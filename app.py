# -*- coding: utf-8 -*-
# app.py — Coordenadas + Δh (ITM/MSAM)
# Permite definir resolución deseada (m/pixel), calcula promedios Δh, h10, h90,
# muestra perfil Plotly y mapa Folium.

import streamlit as st
import pandas as pd
import numpy as np
import math
from io import BytesIO
import time, requests
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from pygeodesy.ellipsoidalVincenty import LatLon
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import srtm

# ---------------- Configuración ----------------
st.set_page_config(page_title="Coordenadas + Δh (ITM)", layout="wide")
st.title("🧭 Coordenadas + 🌄 Δh (ITM / estilo MSAM)")

# Estado persistente
if "categoria" not in st.session_state:
    st.session_state.categoria = "Δh – Rugosidad (ITM)"
if "deltaH_state" not in st.session_state:
    st.session_state.deltaH_state = None

# ---------------- Funciones Geodésicas ----------------
R_EARTH_M = 6371000.0

def destination_point(lat_deg, lon_deg, bearing_deg, distance_m):
    lat1 = math.radians(lat_deg); lon1 = math.radians(lon_deg)
    brng = math.radians(bearing_deg); dr = distance_m / R_EARTH_M
    lat2 = math.asin(math.sin(lat1)*math.cos(dr)+math.cos(lat1)*math.sin(dr)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(dr)*math.cos(lat1),
                             math.cos(dr)-math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), (math.degrees(lon2)+540)%360-180

def decimal_a_gms(grados_decimales,tipo):
    d = {"lat":"N" if grados_decimales>=0 else "S",
         "lon":"E" if grados_decimales>=0 else "W"}[tipo]
    gabs=abs(grados_decimales); g=int(gabs)
    m_dec=(gabs-g)*60; m=int(m_dec); s=(m_dec-m)*60
    return f"{g}° {m}' {s:.6f}\" {d}"

def input_coords():
    col1,col2=st.columns(2)
    with col1: lat=float(st.text_input("Latitud (decimal)",value="8.8066"))
    with col2: lon=float(st.text_input("Longitud (decimal)",value="-82.5403"))
    st.caption(f"⇄ GMS: {decimal_a_gms(lat,'lat')} | {decimal_a_gms(lon,'lon')}")
    return lat,lon

# ---------------- Elevaciones ----------------
@st.cache_resource
def get_srtm_data():
    return srtm.get_data()

def elev_srtm(lats,lons,res_m):
    """Selecciona SRTM1 (30 m) o SRTM3 (90 m) según resolución deseada"""
    data=get_srtm_data()
    vals=[data.get_elevation(la,lo) for la,lo in zip(lats,lons)]
    fuente="SRTM1 (≈30 m)" if res_m<=30 else "SRTM3 (≈90 m)"
    return vals,fuente

class Elev429(Exception): pass

@retry(wait=wait_exponential(multiplier=0.8, min=0.5, max=8),
       stop=stop_after_attempt(5),
       retry=retry_if_exception_type(Elev429))
def elev_open_meteo_chunk(lats,lons):
    base="https://api.open-meteo.com/v1/elevation"
    params={"latitude":",".join([f"{x:.6f}" for x in lats]),
            "longitude":",".join([f"{x:.6f}" for x in lons])}
    r=requests.get(base,params=params,timeout=15)
    if r.status_code==429: raise Elev429()
    r.raise_for_status()
    j=r.json(); return j.get("elevation",[])

def elev_open_meteo(lats,lons):
    vals=[]
    for i in range(0,len(lats),80):
        sub=elev_open_meteo_chunk(lats[i:i+80],lons[i:i+80])
        vals.extend(sub); time.sleep(0.25)
    return [float(v) if v is not None else None for v in vals]

def get_elevations(lats,lons,res_m):
    elev,fuente=elev_srtm(lats,lons,res_m)
    if any(v is None for v in elev):
        try:
            elev2=elev_open_meteo(lats,lons)
            elev=[e if e is not None else elev2[i] for i,e in enumerate(elev)]
            fuente+=" + Open-Meteo (respaldo)"
        except: pass
    return elev,fuente

# ---------------- Cálculo Δh ----------------
def build_profile(lat0,lon0,az,start_km,end_km,step_m):
    dists=list(range(int(start_km*1000),int(end_km*1000)+1,int(step_m)))
    lats,lons=[],[]
    for d in dists:
        la,lo=destination_point(lat0,lon0,az,d)
        lats.append(la); lons.append(lo)
    return dists,lats,lons

def compute_delta_h(elev_list):
    arr=np.array([e for e in elev_list if e is not None])
    if arr.size==0: return None,None,None
    h10=float(np.percentile(arr,90)); h90=float(np.percentile(arr,10))
    return h10-h90,h10,h90

# ---------------- Interfaz ----------------
st.markdown("### 🌄 Δh – Rugosidad (ITM/MSAM)")
lat,lon=input_coords()

c=st.columns(5)
with c[0]:
    az_txt=st.text_input("Azimuts (°)",value="0,45,90,135,180,225,270,315")
with c[1]:
    paso_m=st.number_input("Paso (m)",value=500,min_value=100,step=100)
with c[2]:
    res_m=st.number_input("Resolución deseada (m/píxel)",value=30,min_value=10,step=10)
with c[3]:
    start_km,end_km=10.0,50.0
    st.caption("Tramo fijo: 10–50 km")
with c[4]:
    st.caption("Fuente: SRTM 1/3 + Open-Meteo")

if st.button("Calcular Δh"):
    try: az_list=[float(a.strip()) for a in az_txt.split(",") if a.strip()!=""]
    except: st.error("Revisa los acimuts."); st.stop()

    results=[]; profiles={}; prog=st.progress(0)
    n=len(az_list); fuente_global=None
    for i,az in enumerate(az_list,1):
        dists,lats,lons=build_profile(lat,lon,az,start_km,end_km,paso_m)
        elev,fuente=get_elevations(lats,lons,res_m)
        dh,h10,h90=compute_delta_h(elev)
        row={"Azimut (°)":az,"Δh (m)":round(dh,2) if dh else None,
             "h10 (m, P90)":round(h10,2) if h10 else None,
             "h90 (m, P10)":round(h90,2) if h90 else None,
             "Fuente":fuente}
        profiles[az]=pd.DataFrame({
            "Distancia (km)":[d/1000 for d in dists],
            "Elevación (m)":elev})
        results.append(row); fuente_global=fuente; prog.progress(int(i*100/n))
    df=pd.DataFrame(results).sort_values("Azimut (°)").reset_index(drop=True)

    # Promedios
    valid=df.dropna(subset=["Δh (m)"])
    if not valid.empty:
        prom_dh=valid["Δh (m)"].mean(); prom_h10=valid["h10 (m, P90)"].mean()
        prom_h90=valid["h90 (m, P10)"].mean()
        st.markdown(f"""
        #### Promedios (basados en {len(valid)} radiales)
        • **Δh promedio:** {prom_dh:.2f} m  
        • **h10 promedio:** {prom_h10:.2f} m  
        • **h90 promedio:** {prom_h90:.2f} m  
        • **Fuente usada:** {fuente_global}
        """)

    st.dataframe(df,use_container_width=True)

    # Perfil
    if not df.empty:
        az_sel=st.selectbox("Ver perfil (azimut):",df["Azimut (°)"])
        prof=profiles.get(az_sel)
        if prof is not None:
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=prof["Distancia (km)"],
                                     y=prof["Elevación (m)"],mode="lines",
                                     name=f"Perfil Az {az_sel}°"))
            fig.update_layout(title=f"Perfil de terreno – Az {az_sel}°",
                              xaxis_title="Distancia (km)",
                              yaxis_title="Elevación (m)")
            st.plotly_chart(fig,use_container_width=True)

    # Mapa
    m=folium.Map(location=[lat,lon],zoom_start=8,control_scale=True)
    folium.Marker([lat,lon],tooltip="Transmisor",icon=folium.Icon(color="red")).add_to(m)
    for az,prof in profiles.items():
        pts=[destination_point(lat,lon,az,d*1000) for d in prof["Distancia (km)"]]
        folium.PolyLine(pts,weight=3).add_to(m)
    st.subheader("Mapa de radiales (10–50 km)")
    st_folium(m,width=None,height=520)

    # Descargas
    def df_to_excel_bytes(df):
        from openpyxl import Workbook
        from openpyxl.utils.dataframe import dataframe_to_rows
        wb=Workbook();ws=wb.active;ws.title="Δh_ITM"
        for r in dataframe_to_rows(df,index=False,header=True):ws.append(r)
        ws["G1"]=f"Promedios: Δh={prom_dh:.2f} m, h10={prom_h10:.2f} m, h90={prom_h90:.2f} m"
        out=BytesIO();wb.save(out);return out.getvalue()

    st.download_button("⬇️ CSV",data=df.to_csv(index=False).encode("utf-8"),
                       file_name="DeltaH_ITM.csv",mime="text/csv")
    st.download_button("⬇️ Excel",data=df_to_excel_bytes(df),
                       file_name="DeltaH_ITM.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
