# -*- coding: utf-8 -*-
# app.py — Coordenadas + Δh (ITM/MSAM) 0–50 km
# Incluye:
# - 8 radiales, cálculo por azimut, distancia, distancia central
# - Δh con metodología tipo FCC/MSAM:
#   ordenar elevaciones → percentil 10 → percentil 90 → Δh = h10 – h90
# - Tramo 0–50 km, paso editable
# - SRTM (srtm.py) + Open-Meteo como respaldo
# - Conversión Decimal ↔ GMS
# - Mapas y perfiles interactivos

import streamlit as st
import pandas as pd
import numpy as np
import math
import time
import requests
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

from pygeodesy.ellipsoidalVincenty import LatLon
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import srtm

# ------------------------------------------------------------
# CONFIGURACIÓN GENERAL
# ------------------------------------------------------------

st.set_page_config(page_title="Coordenadas + Δh ITM", layout="wide")
st.title("🧭 Calculadora Avanzada de Coordenadas + 🌄 Δh (ITM / FCC / MSAM)")

# ------------------------------------------------------------
# ESTADOS
# ------------------------------------------------------------

if "categoria" not in st.session_state:
    st.session_state.categoria = "Cálculo - 8 Radiales"

if "resultados" not in st.session_state:
    st.session_state.resultados = {}

if "deltaH_state" not in st.session_state:
    st.session_state.deltaH_state = None

# ------------------------------------------------------------
# FUNCIONES GEO Y CONVERSIONES
# ------------------------------------------------------------

R_EARTH_M = 6371000.0

def destination_point(lat_deg, lon_deg, bearing_deg, distance_m):
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    brng = math.radians(bearing_deg)
    dr = distance_m / R_EARTH_M

    lat2 = math.asin(
        math.sin(lat1)*math.cos(dr) +
        math.cos(lat1)*math.sin(dr)*math.cos(brng)
    )

    lon2 = lon1 + math.atan2(
        math.sin(brng)*math.sin(dr)*math.cos(lat1),
        math.cos(dr) - math.sin(lat1)*math.sin(lat2)
    )

    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180

def decimal_a_gms(dec, tipo):
    d = {"lat": "N" if dec >= 0 else "S",
         "lon": "E" if dec >= 0 else "W"}[tipo]
    v = abs(dec)
    g = int(v)
    m_dec = (v - g) * 60
    m = int(m_dec)
    s = (m_dec - m) * 60
    return f"{g}° {m}' {s:.6f}\" {d}"

def gms_a_decimal(g, m, s, d, tipo):
    dec = abs(g) + m/60 + s/3600
    if d in ("S", "W"):
        dec = -dec
    return dec

# ---------------- INPUT DECIMAL ----------------

def input_decimal(label_lat, label_lon, key_prefix):
    c1, c2 = st.columns(2)
    with c1:
        lat_txt = st.text_input(label_lat, value="8.8066", key=f"{key_prefix}_lat")
    with c2:
        lon_txt = st.text_input(label_lon, value="-82.5403", key=f"{key_prefix}_lon")

    try:
        lat = float(lat_txt)
        lon = float(lon_txt)
    except:
        st.error("Coordenadas decimales inválidas.")
        st.stop()

    st.caption(f"↔ GMS: {decimal_a_gms(lat,'lat')} | {decimal_a_gms(lon,'lon')}")
    return lat, lon

# ---------------- INPUT GMS ----------------

def input_gms(key_prefix):
    st.write("**Latitud (GMS)**")
    a,b,c,d = st.columns(4)
    with a:
        g1 = st.number_input("Grados", value=8, step=1, key=f"{key_prefix}_lat_g")
    with b:
        m1 = st.number_input("Min", value=48, min_value=0, max_value=59, step=1, key=f"{key_prefix}_lat_m")
    with c:
        s1 = st.number_input("Seg", value=23.76, min_value=0.0, max_value=59.999999,
                             step=0.01, key=f"{key_prefix}_lat_s")
    with d:
        d1c = st.selectbox("Dir", ["N","S"], index=0, key=f"{key_prefix}_lat_d")

    st.write("**Longitud (GMS)**")
    e,f,g,h = st.columns(4)
    with e:
        g2 = st.number_input("Grados ", value=82, step=1, key=f"{key_prefix}_lon_g")
    with f:
        m2 = st.number_input("Min ", value=32, min_value=0, max_value=59, step=1, key=f"{key_prefix}_lon_m")
    with g:
        s2 = st.number_input("Seg ", value=25.08, min_value=0.0, max_value=59.999999,
                             step=0.01, key=f"{key_prefix}_lon_s")
    with h:
        d2c = st.selectbox("Dir ", ["E","W"], index=1, key=f"{key_prefix}_lon_d")

    lat = gms_a_decimal(g1, m1, s1, d1c, "lat")
    lon = gms_a_decimal(g2, m2, s2, d2c, "lon")

    st.caption(f"↔ Decimal: {lat:.10f}, {lon:.10f}")
    return lat, lon

# ---------------- SELECCIÓN FORMATO ----------------

def input_coords(key_prefix="base"):
    modo = st.radio(
        "Formato de entrada",
        ["Decimal", "GMS"],
        horizontal=True,
        key=f"{key_prefix}_fmt"
    )

    if modo == "Decimal":
        return input_decimal("Latitud (decimal)", "Longitud (decimal)", key_prefix)
    else:
        return input_gms(key_prefix)

# ------------------------------------------------------------
# ELEVACIONES (SRTM + FALLBACK OPEN-METEO)
# ------------------------------------------------------------

@st.cache_resource
def get_srtm_data():
    return srtm.get_data()

def elev_srtm(lats, lons):
    data = get_srtm_data()
    vals = [data.get_elevation(la,lo) for la,lo in zip(lats,lons)]
    return vals

class Elev429(Exception):
    pass

from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

@retry(wait=wait_exponential(min=0.5, max=4),
       stop=stop_after_attempt(5),
       retry=retry_if_exception_type(Elev429))
def elev_open_meteo_chunk(lats, lons):
    base = "https://api.open-meteo.com/v1/elevation"
    params = {
        "latitude": ",".join([f"{x:.6f}" for x in lats]),
        "longitude": ",".join([f"{x:.6f}" for x in lons])
    }
    r = requests.get(base, params=params, timeout=15)
    if r.status_code == 429:
        raise Elev429()
    r.raise_for_status()
    return r.json().get("elevation", [])

def elev_open_meteo(lats, lons):
    out = []
    for i in range(0, len(lats), 80):
        sub = elev_open_meteo_chunk(lats[i:i+80], lons[i:i+80])
        out.extend(sub)
        time.sleep(0.2)
    return [float(v) if v is not None else None for v in out]

def get_elevations(lats, lons):
    elev = elev_srtm(lats, lons)
    if any(v is None for v in elev):
        try:
            elev2 = elev_open_meteo(lats, lons)
            for i, v in enumerate(elev):
                if v is None:
                    elev[i] = elev2[i]
        except:
            pass
    return elev

# ------------------------------------------------------------
# UTILIDADES DE COORDENADAS
# ------------------------------------------------------------

def calcular_puntos(lat, lon, acimuts, distancias_m):
    base = LatLon(lat, lon)
    out = []
    for d in distancias_m:
        for az in acimuts:
            p = base.destination(d, az)
            out.append({
                "Distancia (km)": d/1000,
                "Acimut (°)": az,
                "Latitud Final": f"{p.lat:.10f}",
                "Longitud Final": f"{p.lon:.10f}",
                "Lat (GMS)": decimal_a_gms(p.lat, "lat"),
                "Lon (GMS)": decimal_a_gms(p.lon, "lon")
            })
    return pd.DataFrame(out)

def calcular_distancia_azimut(lat1, lon1, lat2, lon2):
    p1 = LatLon(lat1, lon1)
    p2 = LatLon(lat2, lon2)
    dkm = p1.distanceTo(p2) / 1000
    az12 = p1.initialBearingTo(p2)
    az21 = p2.initialBearingTo(p1)
    return dkm, az12, az21

def build_profile(lat, lon, az, step_m):
    # Perfil desde 0 hasta 50 km (0–50000 m)
    dists = list(range(0, 50001, step_m))
    lats, lons = [], []
    for d in dists:
        la, lo = destination_point(lat, lon, az, d)
        lats.append(la)
        lons.append(lo)
    return dists, lats, lons

# ------------------------------------------------------------
# METODOLOGÍA PARA Δh (0–50 km)
# ------------------------------------------------------------

def compute_delta_h(elev_list):
    # 1. Filtrar Nones
    # 2. Ordenar elevaciones
    # 3. h10 = percentil 90
    # 4. h90 = percentil 10
    # 5. Δh = h10 – h90
    arr = np.array([e for e in elev_list if e is not None])
    if arr.size == 0:
        return None, None, None
    arr_sorted = np.sort(arr)
    h10 = float(np.percentile(arr_sorted, 90))
    h90 = float(np.percentile(arr_sorted, 10))
    return h10 - h90, h10, h90

# ------------------------------------------------------------
# MENÚ/MOSAICO DE CATEGORÍAS
# ------------------------------------------------------------

st.markdown("### Selecciona una categoría")

c1, c2 = st.columns(2)
c3, c4 = st.columns(2)
c5, _ = st.columns(2)

if c1.button("📍 Cálculo - 8 Radiales"):
    st.session_state.categoria = "Cálculo - 8 Radiales"

if c2.button("🧭 Cálculo por Azimut"):
    st.session_state.categoria = "Cálculo por Azimut"

if c3.button("📏 Cálculo de Distancia"):
    st.session_state.categoria = "Cálculo de Distancia"

if c4.button("🗺️ Cálculo de Distancia Central"):
    st.session_state.categoria = "Cálculo de Distancia Central"

if c5.button("🌄 Δh – Rugosidad"):
    st.session_state.categoria = "Δh – Rugosidad"

categoria = st.session_state.categoria
st.markdown(f"### 🟢 Categoría seleccionada: **{categoria}**")

# ------------------------------------------------------------
# COORDENADAS BASE
# ------------------------------------------------------------

lat, lon = input_coords(key_prefix=f"{categoria}_base")

# ------------------------------------------------------------
# CÁLCULOS SEGÚN CATEGORÍA
# ------------------------------------------------------------

# ------------------- 8 RADIALES -------------------

if categoria == "Cálculo - 8 Radiales":
    acimuts = [0, 45, 90, 135, 180, 225, 270, 315]
    dist_m = [10000, 50000]
    if st.button("Calcular", key="calc8"):
        st.session_state.resultados[categoria] = calcular_puntos(lat, lon, acimuts, dist_m)

# ------------------- CÁLCULO POR AZIMUT -------------------

elif categoria == "Cálculo por Azimut":
    az_txt = st.text_input(
        "Azimuts (°) separados por coma",
        value="0,45,90,135,180,225,270,315"
    )
    d1 = st.number_input("Distancia 1 (m)", value=10000, min_value=1, step=100)
    d2 = st.number_input("Distancia 2 (m)", value=50000, min_value=1, step=100)

    if st.button("Calcular", key="calcaz"):
        try:
            acimuts = [float(a.strip()) for a in az_txt.split(",") if a.strip() != ""]
        except:
            st.error("Error en la lista de azimuts.")
            st.stop()
        st.session_state.resultados[categoria] = calcular_puntos(lat, lon, acimuts, [d1, d2])

# ------------------- DISTANCIA DIRECTA -------------------

elif categoria == "Cálculo de Distancia":
    modo2 = st.radio("Formato punto 2", ["Decimal", "GMS"], horizontal=True)

    if modo2 == "Decimal":
        c1, c2 = st.columns(2)
        with c1:
            lat2 = st.text_input("Latitud 2", value="8.8066")
        with c2:
            lon2 = st.text_input("Longitud 2", value="-82.5403")
        lat2f = float(lat2)
        lon2f = float(lon2)
    else:
        lat2f, lon2f = input_gms("destino")

    if st.button("Calcular", key="calcdist"):
        dkm, az12, az21 = calcular_distancia_azimut(lat, lon, lat2f, lon2f)
        st.session_state.resultados[categoria] = pd.DataFrame([{
            "Distancia (km)": dkm,
            "Acimut ida": az12,
            "Acimut vuelta": az21
        }])

# ------------------- DISTANCIA CENTRAL -------------------

elif categoria == "Cálculo de Distancia Central":
    n = st.number_input("Número de puntos", value=2, min_value=1, step=1)
    filas = []
    for i in range(int(n)):
        modo = st.radio(f"Formato punto {i+1}", ["Decimal", "GMS"], horizontal=True, key=f"fmt_{i}")
        if modo == "Decimal":
            latp = float(st.text_input(f"Latitud punto {i+1}", value="8.8066", key=f"latp_{i}"))
            lonp = float(st.text_input(f"Longitud punto {i+1}", value="-82.5403", key=f"lonp_{i}"))
        else:
            latp, lonp = input_gms(f"p{i}")
        dkm, az12, az21 = calcular_distancia_azimut(lat, lon, latp, lonp)
        filas.append({
            "Punto": i+1,
            "Distancia (km)": dkm,
            "Acimut ida": az12,
            "Acimut vuelta": az21,
        })
    if st.button("Calcular", key="calccentral"):
        st.session_state.resultados[categoria] = pd.DataFrame(filas)

# ------------------------------------------------------------
# Δh – RUGOSIDAD (0–50 km)
# ------------------------------------------------------------

elif categoria == "Δh – Rugosidad":
    az_txt = st.text_input("Azimuts (°)", value="0,45,90,135,180,225,270,315")
    paso_m = st.number_input(
        "Paso (m)",
        value=500,
        min_value=50,
        step=50,
        help="Tramo 0–50 km, separación entre puntos."
    )

    if st.button("Calcular Δh", key="calcdh"):
        try:
            az_list = [float(a.strip()) for a in az_txt.split(",") if a.strip() != ""]
        except:
            st.error("Revisa la lista de azimuts.")
            st.stop()

        results = []
        profiles = []
        profiles_dict = {}

        pb = st.progress(0)
        total = len(az_list)

        for i, az in enumerate(az_list, start=1):
            dists, lats, lons = build_profile(lat, lon, az, paso_m)
            elev = get_elevations(lats, lons)
            dh, h10, h90 = compute_delta_h(elev)

            results.append({
                "Azimut (°)": az,
                "Δh (m)": dh,
                "h10 (P90)": h10,
                "h90 (P10)": h90
            })

            df_prof = pd.DataFrame({
                "Distancia (km)": [d/1000 for d in dists],
                "Elevación (m)": elev
            })
            profiles_dict[az] = df_prof
            pb.progress(int(i*100/total))

        df = pd.DataFrame(results).sort_values("Azimut (°)")

        st.session_state.deltaH_state = {
            "df": df,
            "profiles": profiles_dict,
            "paso": paso_m,
        }

# ------------------------------------------------------------
# RESULTADOS (CUALQUIER CATEGORÍA)
# ------------------------------------------------------------

# ----- PARA LAS CATEGORÍAS NORMALES -----

if categoria in st.session_state.resultados and categoria != "Δh – Rugosidad":
    df = st.session_state.resultados[categoria]
    st.subheader("Resultados")
    st.dataframe(df, use_container_width=True)

    m = folium.Map(location=[lat, lon], zoom_start=9)
    folium.Marker([lat, lon], tooltip="Punto inicial",
                  icon=folium.Icon(color="red")).add_to(m)
    st_folium(m, width=None, height=450)

# ----- PARA RUGOSIDAD Δh -----

if categoria == "Δh – Rugosidad" and st.session_state.deltaH_state:
    data = st.session_state.deltaH_state
    df = data["df"]
    profiles = data["profiles"]

    st.subheader("Resultados de Rugosidad Δh (0–50 km)")
    st.dataframe(df, use_container_width=True)

    if "Δh (m)" in df.columns and df["Δh (m)"].notna().any():
        st.markdown(f"**Δh promedio:** {df['Δh (m)'].mean():.2f} m")

    az_sel = st.selectbox("Ver perfil:", df["Azimut (°)"])
    prof = profiles.get(az_sel)

    if prof is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prof["Distancia (km)"],
            y=prof["Elevación (m)"],
            mode="lines",
            name=f"Perfil {az_sel}°"
        ))
        fig.update_layout(
            title=f"Perfil de Terreno — Azimut {az_sel}° (0–50 km)",
            xaxis_title="Distancia (km)",
            yaxis_title="Elevación (m)"
        )
        st.plotly_chart(fig, use_container_width=True)

    m = folium.Map(location=[lat, lon], zoom_start=8)
    folium.Marker([lat, lon], tooltip="Transmisor",
                  icon=folium.Icon(color="red")).add_to(m)

    for az, prof in profiles.items():
        pts = []
        for dkm in prof["Distancia (km)"]:
            la, lo = destination_point(lat, lon, az, dkm*1000)
            pts.append([la, lo])
        folium.PolyLine(pts, weight=3, opacity=0.85).add_to(m)

    st.subheader("Mapa de Radiales (0–50 km)")
    st_folium(m, width=None, height=520)

    st.download_button(
        "Descargar CSV Δh",
        df.to_csv(index=False).encode("utf-8"),
        "DeltaH_resultados.csv",
        "text/csv"
    )
