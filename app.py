# -*- coding: utf-8 -*-
# app.py — Coordenadas + Δh (ITM/MSAM)
# Incluye:
# - 8 Radiales
# - Por Azimut
# - Distancia
# - Distancia Central
# - Δh (FCC/MSAM con percentiles sobre lista ordenada)
# Elevaciones: SRTM 1/3 (srtm.py) + respaldo Open-Meteo
# Tramo Δh: 10–50 km, paso editable, resolución editable
# Conversión decimal ↔ GMS
# Mapas Folium, perfiles Plotly, descargas CSV/Excel

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

# ---------------- Estado ----------------
if "categoria" not in st.session_state:
    st.session_state.categoria = "Cálculo - 8 Radiales"
if "resultados" not in st.session_state:
    st.session_state.resultados = {}
if "deltaH_state" not in st.session_state:
    st.session_state.deltaH_state = None

# ---------------- Geodesia / conversiones ----------------
R_EARTH_M = 6371000.0

def destination_point(lat_deg, lon_deg, bearing_deg, distance_m):
    lat1 = math.radians(lat_deg); lon1 = math.radians(lon_deg)
    brng = math.radians(bearing_deg); dr = distance_m / R_EARTH_M
    lat2 = math.asin(math.sin(lat1)*math.cos(dr)+math.cos(lat1)*math.sin(dr)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(dr)*math.cos(lat1),
                             math.cos(dr)-math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), (math.degrees(lon2)+540)%360-180

def decimal_a_gms(g,tipo):
    d={"lat":"N" if g>=0 else "S", "lon":"E" if g>=0 else "W"}[tipo]
    gabs=abs(g); g0=int(gabs)
    mdec=(gabs-g0)*60; m=int(mdec)
    s=(mdec-m)*60
    return f"{g0}° {m}' {s:.8f}\" {d}"

def gms_a_decimal(g,m,s,d,tipo):
    dec=abs(g)+m/60+s/3600
    if d in ("S","W"): dec=-dec
    return dec

def input_decimal(label_lat,label_lon,key):
    c1,c2=st.columns(2)
    with c1: lat=float(st.text_input(label_lat,"8.8066",key=f"{key}_lat"))
    with c2: lon=float(st.text_input(label_lon,"-82.5403",key=f"{key}_lon"))
    st.caption(f"GMS → Lat {decimal_a_gms(lat,'lat')} | Lon {decimal_a_gms(lon,'lon')}")
    return lat,lon

def input_gms(key):
    st.write("Latitud (GMS)")
    a,b,c,d = st.columns(4)
    with a: g1=st.number_input("Grados",8)
    with b: m1=st.number_input("Min",48)
    with c: s1=st.number_input("Seg",23.76,step=0.01)
    with d: d1=st.selectbox("Dir",["N","S"])
    st.write("Longitud (GMS)")
    e,f,g,h = st.columns(4)
    with e: g2=st.number_input("Grados_lon",82)
    with f: m2=st.number_input("Min_lon",32)
    with g: s2=st.number_input("Seg_lon",25.08,step=0.01)
    with h: d2=st.selectbox("Dir_lon",["E","W"],index=1)
    lat=gms_a_decimal(g1,m1,s1,d1,"lat")
    lon=gms_a_decimal(g2,m2,s2,d2,"lon")
    st.caption(f"Decimal → Lat {lat} | Lon {lon}")
    return lat,lon

def input_coords(key):
    modo=st.radio("Formato:",["Decimal","GMS"],horizontal=True,key=f"{key}_fmt")
    if modo=="Decimal": return input_decimal("Latitud (decimal)","Longitud (decimal)",key)
    return input_gms(key)

# ---------------- Elevaciones ----------------
@st.cache_resource
def get_srtm_data():
    return srtm.get_data()

def elev_srtm(lats,lons,res):
    data=get_srtm_data()
    vals=[data.get_elevation(a,b) for a,b in zip(lats,lons)]
    fuente = "SRTM1 (30 m)" if res<=30 else "SRTM3 (90 m)"
    return vals,fuente

class Elev429(Exception): pass

@retry(wait=wait_exponential(multiplier=0.8,min=0.5,max=8),
       stop=stop_after_attempt(5),
       retry=retry_if_exception_type(Elev429))
def elev_open_chunk(lat,lon):
    base="https://api.open-meteo.com/v1/elevation"
    r=requests.get(base,params={
        "latitude":",".join([f"{x:.6f}" for x in lat]),
        "longitude":",".join([f"{x:.6f}" for x in lon])
    },timeout=15)
    if r.status_code==429: raise Elev429()
    r.raise_for_status()
    return r.json().get("elevation",[])

def elev_open(lats,lons):
    out=[]
    for i in range(0,len(lats),80):
        sub=elev_open_chunk(lats[i:i+80],lons[i:i+80])
        out.extend(sub); time.sleep(0.2)
    return [float(v) if v is not None else None for v in out]

def get_elevations(lats,lons,res):
    elev,fuente=elev_srtm(lats,lons,res)
    if any(v is None for v in elev):
        try:
            elev2=elev_open(lats,lons)
            for i,v in enumerate(elev):
                elev[i]=v if v is not None else elev2[i]
            fuente += " + Open-Meteo"
        except:
            pass
    return elev,fuente

# ---------------- Perfil y Δh ----------------
def build_profile(lat0,lon0,az,start_km,end_km,step_m):
    d=list(range(int(start_km*1000),int(end_km*1000)+1,int(step_m)))
    LATS=[]; LONS=[]
    for di in d:
        la,lo=destination_point(lat0,lon0,az,di)
        LATS.append(la); LONS.append(lo)
    return d,LATS,LONS

# *** METODOLOGÍA FCC/MSAM — ORDENAR → P10/P90 ***
def compute_delta_h(elev_list):
    arr=np.array([e for e in elev_list if e is not None],float)
    if arr.size<5: return None,None,None

    # 1. Orden explícito
    elev_sorted=np.sort(arr)

    # 2. Percentiles FCC
    h10=float(np.percentile(elev_sorted,90))
    h90=float(np.percentile(elev_sorted,10))

    # 3. Δh
    dh=h10-h90
    return round(dh,2),round(h10,2),round(h90,2)

# ---------------- Utilidades varias ----------------
def calcular_puntos(lat,lon,azs,dists):
    base=LatLon(lat,lon); out=[]
    for d in dists:
        for az in azs:
            p=base.destination(d,az)
            out.append({
                "Distancia (km)": d/1000,
                "Acimut (°)": az,
                "Latitud Final (Decimal)": f"{p.lat:.10f}",
                "Longitud Final (Decimal)": f"{p.lon:.10f}",
                "Latitud (GMS)": decimal_a_gms(p.lat,"lat"),
                "Longitud (GMS)": decimal_a_gms(p.lon,"lon")
            })
    return pd.DataFrame(out)

def calcular_distancia_azimut(lat1,lon1,lat2,lon2):
    p1=LatLon(lat1,lon1); p2=LatLon(lat2,lon2)
    d=p1.distanceTo(p2)/1000
    return d,p1.initialBearingTo(p2),p2.initialBearingTo(p1)

def mostrar_mapa(df,lat,lon,categoria):
    m=folium.Map(location=[lat,lon],zoom_start=9,control_scale=True)
    if categoria in ("Cálculo - 8 Radiales","Cálculo por Azimut"):
        for _,r in df.iterrows():
            folium.Marker([float(r["Latitud Final (Decimal)"]),
                           float(r["Longitud Final (Decimal)"])],
                           tooltip=f"{r.get('Acimut (°)','')}°").add_to(m)
        folium.Marker([lat,lon],icon=folium.Icon(color="red")).add_to(m)
    st_folium(m,width=None,height=450)

# ---------------- Mosaico ----------------
st.markdown("### Selecciona la categoría:")
c1,c2 = st.columns(2)
c3,c4 = st.columns(2)
c5,_ = st.columns(2)

if c1.button("📍 Cálculo - 8 Radiales"): st.session_state.categoria="Cálculo - 8 Radiales"
if c2.button("🧭 Cálculo por Azimut"):   st.session_state.categoria="Cálculo por Azimut"
if c3.button("📏 Cálculo de Distancia"): st.session_state.categoria="Cálculo de Distancia"
if c4.button("🗺️ Cálculo Distancia Central"): st.session_state.categoria="Cálculo de Distancia Central"
if c5.button("🌄 Δh – Rugosidad (ITM)"): st.session_state.categoria="Δh – Rugosidad (ITM)"

categoria=st.session_state.categoria
st.markdown(f"### 🟢 Categoría seleccionada: {categoria}")

# ---------------- Coordenadas base ----------------
lat,lon=input_coords(categoria)

# ---------------- Cálculos ----------------
if categoria=="Cálculo - 8 Radiales":
    if st.button("Calcular"):
        st.session_state.resultados[categoria]=calcular_puntos(lat,lon,[0,45,90,135,180,225,270,315],[10000,50000])

elif categoria=="Cálculo por Azimut":
    az_txt=st.text_input("Azimuts","0,45,90,135,180,225,270,315")
    d1=st.number_input("Distancia 1",10000); d2=st.number_input("Distancia 2",50000)
    if st.button("Calcular"):
        az=[float(a) for a in az_txt.split(",")]
        st.session_state.resultados[categoria]=calcular_puntos(lat,lon,az,[d1,d2])

elif categoria=="Cálculo de Distancia":
    lat2=float(st.text_input("Lat2","8.8066"))
    lon2=float(st.text_input("Lon2","-82.5403"))
    if st.button("Calcular"):
        d,az1,az2=calcular_distancia_azimut(lat,lon,lat2,lon2)
        st.session_state.resultados[categoria]=pd.DataFrame([{
            "Distancia (km)":d,"Acimut ida":az1,"Acimut vuelta":az2
        }])

elif categoria=="Cálculo de Distancia Central":
    n=st.number_input("Número de puntos",2)
    filas=[]
    for i in range(int(n)):
        latp=float(st.text_input(f"Latitud punto {i+1}","8.8066"))
        lonp=float(st.text_input(f"Longitud punto {i+1}","-82.5403"))
        d,az1,az2=calcular_distancia_azimut(lat,lon,latp,lonp)
        filas.append({"Distancia":d,"Acimut ida":az1,"Acimut vuelta":az2})
    if st.button("Calcular"):
        st.session_state.resultados[categoria]=pd.DataFrame(filas)

# ---------------- Δh – Rugosidad (ITM) ----------------
if categoria=="Δh – Rugosidad (ITM)":

    az_txt=st.text_input("Azimuts","0,45,90,135,180,225,270,315")
    paso=st.number_input("Paso (m)",500)
    res=st.number_input("Resolución deseada (m/pixel)",30,min_value=10)

    if st.button("Calcular Δh"):
        az_list=[float(a) for a in az_txt.split(",")]
        results=[] ; profiles={}
        start_km,end_km=10,50
        prog=st.progress(0)

        for i,az in enumerate(az_list,1):
            d,lats,lons=build_profile(lat,lon,az,start_km,end_km,paso)
            elev,fuente=get_elevations(lats,lons,res)
            dh,h10,h90=compute_delta_h(elev)
            results.append({
                "Azimut (°)":az,
                "Δh (m)":dh,
                "h10 (P90)":h10,
                "h90 (P10)":h90,
                "Fuente":fuente
            })
            profiles[az]=pd.DataFrame({"Dist(km)":[x/1000 for x in d],"Elev(m)":elev})
            prog.progress(int(i*100/len(az_list)))

        df=pd.DataFrame(results)

        # promedios
        v=df.dropna(subset=["Δh (m)"])
        prom_dh=v["Δh (m)"].mean()
        prom_h10=v["h10 (P90)"].mean()
        prom_h90=v["h90 (P10)"].mean()

        st.subheader("Resultados Δh")
        st.markdown(f"""
**Promedios FCC/MSAM**
- Δh promedio: **{prom_dh:.2f} m**
- h10 promedio: **{prom_h10:.2f} m**
- h90 promedio: **{prom_h90:.2f} m**
- Resolución usada: **{res} m/pixel**
        """)

        st.dataframe(df,use_container_width=True)

        # Perfil
        azsel=st.selectbox("Perfil:",df["Azimut (°)"])
        pr=profiles[azsel]
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=pr["Dist(km)"],y=pr["Elev(m)"],mode="lines"))
        fig.update_layout(title=f"Perfil Az {azsel}°",xaxis_title="km",yaxis_title="Elev (m)")
        st.plotly_chart(fig,use_container_width=True)

        # Mapa
        m=folium.Map(location=[lat,lon],zoom_start=8)
        folium.Marker([lat,lon],icon=folium.Icon(color="red")).add_to(m)
        for az in az_list:
            pts=[]
            prof=profiles[az]
            for dkm in prof["Dist(km)"]:
                la,lo=destination_point(lat,lon,az,dkm*1000)
                pts.append([la,lo])
            folium.PolyLine(pts,weight=3).add_to(m)
        st_folium(m,width=None,height=520)

        # Descargas
        st.download_button("CSV Δh",data=df.to_csv(index=False).encode("utf-8"),
                           file_name="DeltaH_ITM.csv",mime="text/csv")
