# -*- coding: utf-8 -*-
# app.py — Coordenadas + Δh (ITM/MSAM) completo
# - Pestañas: 8 Radiales, Por Azimut, Distancia, Distancia Central, Δh (ITM)
# - Δh: 10–50 km, paso editable (default 500 m), m/píxel deseados (prioriza SRTM1≈30 m)
# - Elevaciones: SRTM (srtm.py) + respaldo Open-Meteo en lotes con reintentos
# - Conversión Decimal↔GMS, resultados persistentes, perfil Plotly, mapa Folium, descargas CSV/Excel

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
    st.session_state.deltaH_state = None  # dict: inputs, df, profiles, promedios

# ---------------- Geodesia / conversiones ----------------
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
    return f"{g}° {m}' {s:.8f}\" {d}"

def gms_a_decimal(grados:int, minutos:int, segundos:float, direccion:str, tipo:str):
    if tipo=="lat" and not (0<=abs(grados)<=90):  raise ValueError("Lat 0–90")
    if tipo=="lon" and not (0<=abs(grados)<=180): raise ValueError("Lon 0–180")
    if not (0<=minutos<60):  raise ValueError("Min 0–59")
    if not (0<=segundos<60): raise ValueError("Seg 0–59.999")
    if tipo=="lat" and direccion not in ("N","S"): raise ValueError("Dir N/S")
    if tipo=="lon" and direccion not in ("E","W"): raise ValueError("Dir E/W")
    dec = abs(grados)+minutos/60+segundos/3600
    if direccion in ("S","W"): dec = -dec
    return dec

def input_decimal(label_lat, label_lon, key_prefix):
    c1,c2 = st.columns(2)
    with c1: lat_txt = st.text_input(label_lat, value="8.8066", key=f"{key_prefix}_lat_dec")
    with c2: lon_txt = st.text_input(label_lon, value="-82.5403", key=f"{key_prefix}_lon_dec")
    try:
        lat=float(lat_txt); lon=float(lon_txt)
    except ValueError:
        st.error("Lat/Lon decimales inválidos."); st.stop()
    st.caption(f"⇄ GMS: Lat {decimal_a_gms(lat,'lat')} | Lon {decimal_a_gms(lon,'lon')}")
    return lat, lon

def input_gms(key_prefix, defaults=("N","W")):
    st.write("**Latitud (GMS)**")
    a,b,c,d = st.columns([1,1,1,1])
    with a: lat_g=st.number_input("Grados", value=8, step=1, format="%d", key=f"{key_prefix}_lat_g")
    with b: lat_m=st.number_input("Min", value=48, min_value=0, max_value=59, step=1, format="%d", key=f"{key_prefix}_lat_m")
    with c: lat_s=st.number_input("Seg", value=23.76, min_value=0.0, max_value=59.999999, step=0.01, format="%.6f", key=f"{key_prefix}_lat_s")
    with d: lat_d=st.selectbox("Dir", ["N","S"], index=0 if defaults[0]=="N" else 1, key=f"{key_prefix}_lat_d")

    st.write("**Longitud (GMS)**")
    e,f,g,h = st.columns([1,1,1,1])
    with e: lon_g=st.number_input("Grados", value=82, step=1, format="%d", key=f"{key_prefix}_lon_g")
    with f: lon_m=st.number_input("Min", value=32, min_value=0, max_value=59, step=1, format="%d", key=f"{key_prefix}_lon_m")
    with g: lon_s=st.number_input("Seg", value=25.08, min_value=0.0, max_value=59.999999, step=0.01, format="%.6f", key=f"{key_prefix}_lon_s")
    with h: lon_d=st.selectbox("Dir", ["E","W"], index=1 if defaults[1]=="W" else 0, key=f"{key_prefix}_lon_d")

    lat=gms_a_decimal(lat_g,lat_m,lat_s,lat_d,"lat")
    lon=gms_a_decimal(lon_g,lon_m,lon_s,lon_d,"lon")
    st.caption(f"⇄ Decimal: Lat {lat:.10f} | Lon {lon:.10f}")
    return lat, lon

def input_coords(key_prefix="base"):
    st.markdown("#### Formato de coordenadas de entrada")
    modo = st.radio("Formato", ["Decimal", "Grados, Minutos y Segundos (GMS)"], horizontal=True, key=f"{key_prefix}_fmt")
    if modo=="Decimal":
        return input_decimal("Latitud (decimal)", "Longitud (decimal)", key_prefix)
    return input_gms(key_prefix)

# ---------------- Elevaciones ----------------
@st.cache_resource
def get_srtm_data():
    return srtm.get_data()  # SRTM1≈30 m donde disponible; si no, SRTM3≈90 m

def elev_srtm(lats,lons,res_mpx):
    data=get_srtm_data()
    vals=[data.get_elevation(la,lo) for la,lo in zip(lats,lons)]
    fuente="SRTM1 (≈30 m)" if res_mpx<=30 else "SRTM3 (≈90 m)"
    return vals, fuente

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
    out=[]
    for i in range(0,len(lats),80):
        sub=elev_open_meteo_chunk(lats[i:i+80],lons[i:i+80])
        out.extend(sub); time.sleep(0.25)
    return [float(v) if v is not None else None for v in out]

def get_elevations(lats,lons,res_mpx):
    elev,fuente = elev_srtm(lats,lons,res_mpx)
    if any(v is None for v in elev):
        try:
            elev2=elev_open_meteo(lats,lons)
            for i,v in enumerate(elev):
                elev[i]=v if v is not None else elev2[i]
            fuente += " + Open-Meteo (respaldo)"
        except:
            pass
    return elev, fuente

# ---------------- Utilidades comunes ----------------
def calcular_puntos(lat, lon, acimuts, distancias_m):
    base = LatLon(lat, lon); out=[]
    for d in distancias_m:
        for az in acimuts:
            p=base.destination(d,az)
            out.append({
                "Distancia (km)": d/1000,
                "Acimut (°)": az,
                "Latitud Final (Decimal)": f"{p.lat:.10f}",
                "Longitud Final (Decimal)": f"{p.lon:.10f}",
                "Latitud (GMS)": decimal_a_gms(p.lat, "lat"),
                "Longitud (GMS)": decimal_a_gms(p.lon, "lon")
            })
    return pd.DataFrame(out)

def calcular_distancia_azimut(lat1, lon1, lat2, lon2):
    p1=LatLon(lat1,lon1); p2=LatLon(lat2,lon2)
    d=p1.distanceTo(p2)/1000.0
    az12=p1.initialBearingTo(p2); az21=p2.initialBearingTo(p1)
    return d, az12, az21

def build_profile(lat0, lon0, az, start_km, end_km, step_m):
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
    return h10-h90, h10, h90

def mostrar_mapa_generico(df, lat, lon, categoria):
    m=folium.Map(location=[lat,lon],zoom_start=9,control_scale=True)
    if categoria in ("Cálculo - 8 Radiales","Cálculo por Azimut"):
        for _,r in df.iterrows():
            folium.Marker([float(r["Latitud Final (Decimal)"]),
                           float(r["Longitud Final (Decimal)"])],
                          tooltip=f"{r.get('Acimut (°)','')}° - {r.get('Distancia (km)','')} km").add_to(m)
        folium.Marker([lat,lon],tooltip="Punto inicial",icon=folium.Icon(color="red")).add_to(m)
    elif categoria=="Cálculo de Distancia":
        for _,r in df.iterrows():
            lat2,lon2=float(r["Latitud 2"]),float(r["Longitud 2"])
            folium.Marker([lat,lon],tooltip="Punto 1",icon=folium.Icon(color="red")).add_to(m)
            folium.Marker([lat2,lon2],tooltip="Punto 2",icon=folium.Icon(color="blue")).add_to(m)
            folium.PolyLine([[lat,lon],[lat2,lon2]],weight=2).add_to(m)
    elif categoria=="Cálculo de Distancia Central":
        for _,r in df.iterrows():
            latc,lonc=float(r["Latitud central"]),float(r["Longitud central"])
            latp,lonp=float(r["Latitud punto"]),float(r["Longitud punto"])
            folium.Marker([latc,lonc],tooltip="Central",icon=folium.Icon(color="red")).add_to(m)
            folium.Marker([latp,lonp],tooltip="Punto",icon=folium.Icon(color="blue")).add_to(m)
            folium.PolyLine([[latc,lonc],[latp,lonp]],color="green",weight=2).add_to(m)
    st_folium(m,width=None,height=480)

# ---------------- Mosaico de categorías ----------------
st.markdown("### Selecciona la categoría de cálculo")
c1,c2 = st.columns(2)
c3,c4 = st.columns(2)
c5,_  = st.columns(2)

if c1.button("📍 Cálculo - 8 Radiales"):
    st.session_state.categoria="Cálculo - 8 Radiales"
if c2.button("🧭 Cálculo por Azimut"):
    st.session_state.categoria="Cálculo por Azimut"
if c3.button("📏 Cálculo de Distancia"):
    st.session_state.categoria="Cálculo de Distancia"
if c4.button("🗺️ Cálculo de Distancia Central"):
    st.session_state.categoria="Cálculo de Distancia Central"
if c5.button("🌄 Δh – Rugosidad (ITM)"):
    st.session_state.categoria="Δh – Rugosidad (ITM)"

categoria = st.session_state.categoria
st.markdown(f"### 🟢 Categoría seleccionada: {categoria}")

# ---------------- Coordenadas base ----------------
lat, lon = input_coords(key_prefix=f"{categoria}_base")

# ---------------- Pestañas de coordenadas ----------------
if categoria=="Cálculo - 8 Radiales":
    acimuts=[0,45,90,135,180,225,270,315]
    dist_m=[10000,50000]
    if st.button("Calcular", key="calc_8rad"):
        st.session_state.resultados[categoria]=calcular_puntos(lat,lon,acimuts,dist_m)

elif categoria=="Cálculo por Azimut":
    az_txt=st.text_input("Azimuts (°) separados por coma",value="0,45,90,135,180,225,270,315")
    d1=st.number_input("Distancia 1 (m)",value=10000,min_value=1,step=100)
    d2=st.number_input("Distancia 2 (m)",value=50000,min_value=1,step=100)
    if st.button("Calcular", key="calc_az"):
        try:
            acimuts=[float(a.strip()) for a in az_txt.split(",") if a.strip()!=""]
            st.session_state.resultados[categoria]=calcular_puntos(lat,lon,acimuts,[d1,d2])
        except Exception as e:
            st.error(f"Error en azimuts: {e}")

elif categoria=="Cálculo de Distancia":
    modo2=st.radio("Formato para el Punto 2",["Decimal","GMS"],horizontal=True,key="fmt_p2")
    if modo2=="Decimal":
        c1,c2=st.columns(2)
        with c1: lat2=st.text_input("Latitud 2 (decimal)",value="8.8066")
        with c2: lon2=st.text_input("Longitud 2 (decimal)",value="-82.5403")
        try:
            lat2f=float(lat2); lon2f=float(lon2)
        except ValueError:
            st.error("Lat/Lon decimales inválidos."); st.stop()
        st.caption(f"Punto 2 (GMS): Lat {decimal_a_gms(lat2f,'lat')} | Lon {decimal_a_gms(lon2f,'lon')}")
    else:
        # Entrada GMS del punto 2
        st.write("**Latitud 2 (GMS)**")
        a,b,c,d = st.columns([1,1,1,1])
        with a: g1=st.number_input("Grados", value=8, step=1, format="%d", key="lat2_g")
        with b: m1=st.number_input("Min", value=48, min_value=0, max_value=59, step=1, format="%d", key="lat2_m")
        with c: s1=st.number_input("Seg", value=23.76, min_value=0.0, max_value=59.999999, step=0.01, format="%.6f", key="lat2_s")
        with d: d1c=st.selectbox("Dir", ["N","S"], index=0, key="lat2_d")
        st.write("**Longitud 2 (GMS)**")
        e,f,g,h = st.columns([1,1,1,1])
        with e: g2=st.number_input("Grados", value=82, step=1, format="%d", key="lon2_g")
        with f: m2=st.number_input("Min", value=32, min_value=0, max_value=59, step=1, format="%d", key="lon2_m")
        with g: s2=st.number_input("Seg", value=25.08, min_value=0.0, max_value=59.999999, step=0.01, format="%.6f", key="lon2_s")
        with h: d2c=st.selectbox("Dir", ["E","W"], index=1, key="lon2_d")
        lat2f=gms_a_decimal(g1,m1,s1,d1c,"lat"); lon2f=gms_a_decimal(g2,m2,s2,d2c,"lon")
        st.caption(f"Punto 2 (Decimal): Lat {lat2f:.10f} | Lon {lon2f:.10f}")

    if st.button("Calcular", key="calc_dist"):
        dkm,az12,az21=calcular_distancia_azimut(lat,lon,lat2f,lon2f)
        st.session_state.resultados[categoria]=pd.DataFrame([{
            "Distancia (km)": dkm,
            "Acimut ida (°)": az12,
            "Acimut vuelta (°)": az21,
            "Latitud 1": lat, "Longitud 1": lon,
            "Latitud 2": lat2f, "Longitud 2": lon2f
        }])

elif categoria=="Cálculo de Distancia Central":
    n=st.number_input("Número de puntos",min_value=1,value=2,step=1)
    filas=[]
    for i in range(int(n)):
        modo_i=st.radio(f"Formato Punto {i+1}",["Decimal","GMS"],horizontal=True,key=f"fmt_central_{i}")
        if modo_i=="Decimal":
            c1,c2=st.columns(2)
            with c1: latp=st.text_input(f"Latitud punto {i+1} (decimal)",value="8.8066",key=f"latp_{i}")
            with c2: lonp=st.text_input(f"Longitud punto {i+1} (decimal)",value="-82.5403",key=f"lonp_{i}")
            try: latpf=float(latp); lonpf=float(lonp)
            except ValueError: st.error(f"Punto {i+1}: decimales inválidos."); st.stop()
            st.caption(f"Punto {i+1} (GMS): Lat {decimal_a_gms(latpf,'lat')} | Lon {decimal_a_gms(lonpf,'lon')}")
        else:
            st.write(f"**Latitud punto {i+1} (GMS)**")
            a,b,c,d = st.columns([1,1,1,1])
            with a: g1=st.number_input("Grados", value=8, step=1, format="%d", key=f"latp{i}_g")
            with b: m1=st.number_input("Min", value=48, min_value=0, max_value=59, step=1, format="%d", key=f"latp{i}_m")
            with c: s1=st.number_input("Seg", value=23.76, min_value=0.0, max_value=59.999999, step=0.01, format="%.6f", key=f"latp{i}_s")
            with d: d1c=st.selectbox("Dir", ["N","S"], index=0, key=f"latp{i}_d")
            st.write(f"**Longitud punto {i+1} (GMS)**")
            e,f,g,h = st.columns([1,1,1,1])
            with e: g2=st.number_input("Grados", value=82, step=1, format="%d", key=f"lonp{i}_g")
            with f: m2=st.number_input("Min", value=32, min_value=0, max_value=59, step=1, format="%d", key=f"lonp{i}_m")
            with g: s2=st.number_input("Seg", value=25.08, min_value=0.0, max_value=59.999999, step=0.01, format="%.6f", key=f"lonp{i}_s")
            with h: d2c=st.selectbox("Dir", ["E","W"], index=1, key=f"lonp{i}_d")
            latpf=gms_a_decimal(g1,m1,s1,d1c,"lat"); lonpf=gms_a_decimal(g2,m2,s2,d2c,"lon")
            st.caption(f"Punto {i+1} (Decimal): Lat {latpf:.10f} | Lon {lonpf:.10f}")
        dkm,az12,az21=calcular_distancia_azimut(lat,lon,latpf,lonpf)
        filas.append({
            "Distancia (km)": dkm,
            "Acimut ida (°)": az12,
            "Acimut vuelta (°)": az21,
            "Latitud central": lat, "Longitud central": lon,
            "Latitud punto": latpf, "Longitud punto": lonpf
        })
    if st.button("Calcular", key="calc_central"):
        st.session_state.resultados[categoria]=pd.DataFrame(filas)

# ---------------- Δh – Rugosidad (ITM) ----------------
if categoria=="Δh – Rugosidad (ITM)":
    st.markdown("#### Parámetros Δh (ITM/MSAM)")
    c = st.columns(5)
    with c[0]:
        az_txt = st.text_input("Azimuts (°)", value="0,45,90,135,180,225,270,315")
    with c[1]:
        paso_m = st.number_input("Paso (m)", value=500, min_value=100, step=100, help="Default 500 m (MSAM).")
    with c[2]:
        res_mpx = st.number_input("Resolución deseada (m/píxel)", value=30, min_value=10, step=10,
                                  help="≤30 fuerza SRTM1 (≈30 m), >30 usa SRTM3 (≈90 m).")
    with c[3]:
        st.caption("Tramo fijo: 10–50 km")
    with c[4]:
        st.caption("Fuente: SRTM (1\"/3\") con respaldo Open-Meteo")

    if st.button("Calcular Δh", key="calc_dh_btn"):
        try:
            az_list=[float(a.strip()) for a in az_txt.split(",") if a.strip()!=""]
        except:
            st.error("Revisa los acimuts (usa coma como separador)."); st.stop()

        start_km,end_km = 10.0, 50.0
        results=[]; profiles={}; prog=st.progress(0); n=len(az_list)
        fuente_global=None

        for i,az in enumerate(az_list, start=1):
            dists,lats,lons = build_profile(lat,lon,az,start_km,end_km,paso_m)
            elev, fuente = get_elevations(lats,lons,res_mpx)
            dh,h10,h90 = compute_delta_h(elev)
            row = {
                "Azimut (°)": az,
                "Δh (m)": round(dh,2) if dh is not None else None,
                "h10 (m, P90)": round(h10,2) if h10 is not None else None,
                "h90 (m, P10)": round(h90,2) if h90 is not None else None,
                "Fuente": fuente
            }
            profiles[az]=pd.DataFrame({
                "Distancia (km)": [d/1000 for d in dists],
                "Elevación (m)": elev
            })
            results.append(row); fuente_global=fuente; prog.progress(int(i*100/n))

        df = pd.DataFrame(results).sort_values("Azimut (°)").reset_index(drop=True)

        # Promedios estilo MSAM
        valid = df.dropna(subset=["Δh (m)"])
        prom_dh = valid["Δh (m)"].mean() if not valid.empty else None
        prom_h10 = valid["h10 (m, P90)"].mean() if not valid.empty else None
        prom_h90 = valid["h90 (m, P10)"].mean() if not valid.empty else None

        st.session_state.deltaH_state = {
            "status":"done",
            "inputs":{"azimuts":az_list,"paso_m":paso_m,"tramo_km":(start_km,end_km),"res_mpx":res_mpx},
            "df":df,
            "profiles":profiles,
            "promedios":{"Δh_prom":prom_dh,"h10_prom":prom_h10,"h90_prom":prom_h90,"fuente":fuente_global}
        }

    # Mostrar resultados persistentes
    if st.session_state.deltaH_state and st.session_state.deltaH_state.get("status")=="done":
        state=st.session_state.deltaH_state
        df=state["df"]; profiles=state["profiles"]; prom=state["promedios"]
        paso=state["inputs"]["paso_m"]; tramo=state["inputs"]["tramo_km"]; resmpx=state["inputs"]["res_mpx"]

        st.subheader("Resultados Δh por azimut (ITM/MSAM)")
        if prom["Δh_prom"] is not None:
            st.markdown(f"""
**Promedios (basados en {len(df.dropna(subset=['Δh (m)']))} radiales)**
- Δh promedio: **{prom['Δh_prom']:.2f} m**
- h10 promedio: **{prom['h10_prom']:.2f} m**
- h90 promedio: **{prom['h90_prom']:.2f} m**
- Resolución solicitada: **{resmpx} m/píxel** — Fuente usada: **{prom['fuente']}**
            """)
        st.dataframe(df, use_container_width=True)

        # Perfil interactivo
        if not df.empty:
            az_sel = st.selectbox("Ver perfil (azimut):", df["Azimut (°)"])
            prof = profiles.get(az_sel)
            if prof is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=prof["Distancia (km)"], y=prof["Elevación (m)"], mode="lines",
                                         name=f"Perfil – Az {az_sel}°"))
                fig.update_layout(title=f"Perfil – Az {az_sel}° (Tramo {tramo[0]}–{tramo[1]} km, Paso {paso} m)",
                                  xaxis_title="Distancia (km)", yaxis_title="Elevación (m)")
                st.plotly_chart(fig, use_container_width=True)

        # Mapa de radiales
        m=folium.Map(location=[lat,lon],zoom_start=8,control_scale=True)
        folium.Marker([lat,lon],tooltip="Transmisor",icon=folium.Icon(color="red")).add_to(m)
        for az,prof in profiles.items():
            pts=[]
            for dkm in prof["Distancia (km)"]:
                la,lo = destination_point(lat,lon,az,dkm*1000)
                pts.append([la,lo])
            folium.PolyLine(pts,weight=3,opacity=0.85).add_to(m)
        st.subheader("Mapa de radiales (10–50 km)")
        st_folium(m,width=None,height=520)

        # Descargas
        def df_to_excel_bytes(df, prom):
            from openpyxl import Workbook
            from openpyxl.utils.dataframe import dataframe_to_rows
            wb=Workbook(); ws=wb.active; ws.title="Δh_ITM"
            for r in dataframe_to_rows(df,index=False,header=True): ws.append(r)
            ws["G1"]=f"Promedios: Δh={prom['Δh_prom']:.2f} m; h10={prom['h10_prom']:.2f} m; h90={prom['h90_prom']:.2f} m"
            out=BytesIO(); wb.save(out); return out.getvalue()

        st.download_button("⬇️ CSV (Δh)", data=df.to_csv(index=False).encode("utf-8"),
                           file_name="DeltaH_ITM_resultados.csv", mime="text/csv")
        st.download_button("⬇️ Excel (Δh)", data=df_to_excel_bytes(df, prom),
                           file_name="DeltaH_ITM_resultados.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ---------------- Mostrar resultados de otras categorías ----------------
if categoria in st.session_state.resultados and categoria != "Δh – Rugosidad (ITM)":
    df = st.session_state.resultados[categoria]
    st.subheader("Resultados")
    if "Distancia (km)" in df.columns and categoria in ("Cálculo - 8 Radiales","Cálculo por Azimut"):
        for d in sorted(df["Distancia (km)"].unique()):
            st.markdown(f"**Resultados a {d} km**")
            st.dataframe(df[df["Distancia (km)"]==d], use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

    mostrar_mapa_generico(df, lat, lon, categoria)

    st.download_button(
        "📥 Descargar CSV",
        data=df.to_csv(index=False, sep=';', encoding='utf-8'),
        file_name=f"{categoria.replace(' ','_')}.csv",
        mime="text/csv"
    )
