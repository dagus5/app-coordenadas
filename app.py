# -*- coding: utf-8 -*-
# app.py — Coordenadas + Δh (ITM/FCC/MSAM) 0–50 km
# Incluye:
# - 8 radiales, cálculo por azimut, distancia, distancia central
# - Δh con metodología tipo ITM/FCC/MSAM:
#   ordenar elevaciones → percentiles 10 y 90 → Δh = h90 – h10
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
    
if "fcc_state" not in st.session_state:
    st.session_state.fcc_state = None

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
# METODOLOGÍA PARA Δh (HÍBRIDA ITM/FCC/MSAM + PERSONALIZADO)
# ------------------------------------------------------------

def compute_delta_h(dists_m, elev_list, metodo, d_min_custom=None, d_max_custom=None):
    """
    Cálculo híbrido de Δh:
    - ITM / MSAM: 10–50 km
    - FCC: 3–16 km
    - 0–50 km completo: todo el perfil
    - Personalizado (km): rango definido por el usuario
    Δh = h90 – h10 (percentiles 90 y 10 de la elevación).
    """

    if metodo == "ITM / MSAM (10–50 km)":
        d_min, d_max = 10000, 50000
    elif metodo == "FCC (3–16 km)":
        d_min, d_max = 3000, 16000
    elif metodo == "0–50 km completo":
        d_min, d_max = 0, 50000
    elif metodo == "Personalizado (km)" and d_min_custom is not None and d_max_custom is not None:
        d_min, d_max = d_min_custom, d_max_custom
    else:
        d_min, d_max = 0, max(dists_m) if len(dists_m) > 0 else 0

    # Filtrar por rango de distancias y descartar None
    elev_filtradas = [
        e for d, e in zip(dists_m, elev_list)
        if e is not None and d_min <= d <= d_max
    ]

    if len(elev_filtradas) == 0:
        return None, None, None

    arr = np.sort(np.array(elev_filtradas, dtype=float))
    h10 = float(np.percentile(arr, 10))
    h90 = float(np.percentile(arr, 90))
    delta_h = h90 - h10  # P90 – P10

    return delta_h, h10, h90

RANGO_METODO = {
    "ITM / MSAM (10–50 km)": "10–50",
    "FCC (3–16 km)": "3–16",
    "0–50 km completo": "0–50",
}
@st.cache_data
def cargar_curvas_fcc():
    return pd.read_csv("fcc_fm_f5050.csv")

# ------------------------------------------------------------
# MENÚ/MOSAICO DE CATEGORÍAS
# ------------------------------------------------------------

st.markdown("### Selecciona una categoría")

c1, c2 = st.columns(2)
c3, c4 = st.columns(2)
c5, c6 = st.columns(2)

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
    
if c6.button("📡 Contorno FCC"):
    st.session_state.categoria = "Contorno FCC"

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
# Δh – RUGOSIDAD (0–50 km, HÍBRIDO + PERSONALIZADO)
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

    metodo_dh = st.selectbox(
        "Método de cálculo de Δh",
        ["ITM / MSAM (10–50 km)",
         "FCC (3–16 km)",
         "0–50 km completo",
         "Personalizado (km)"],
        index=0,
        help="Selecciona el rango de distancias sobre el que se calcula la rugosidad."
    )

    d_min_km = None
    d_max_km = None

    if metodo_dh == "Personalizado (km)":
        c1, c2 = st.columns(2)
        with c1:
            d_min_km = st.number_input(
                "Distancia mínima (km)",
                value=5.0,
                min_value=0.0,
                step=0.5
            )
        with c2:
            d_max_km = st.number_input(
                "Distancia máxima (km)",
                value=30.0,
                min_value=0.0,
                step=0.5
            )
        if d_max_km <= d_min_km:
            st.warning("La distancia máxima debe ser mayor que la mínima.")

    if st.button("Calcular Δh", key="calcdh"):
        try:
            az_list = [float(a.strip()) for a in az_txt.split(",") if a.strip() != ""]
        except:
            st.error("Revisa la lista de azimuts.")
            st.stop()

        if metodo_dh == "Personalizado (km)" and (d_min_km is None or d_max_km is None or d_max_km <= d_min_km):
            st.error("Revisa el rango personalizado de distancias (km).")
            st.stop()

        results = []
        profiles_dict = {}

        pb = st.progress(0)
        total = len(az_list)

        # Preconvertir rango personalizado a metros
        d_min_custom_m = d_max_custom_m = None
        if metodo_dh == "Personalizado (km)":
            d_min_custom_m = d_min_km * 1000.0
            d_max_custom_m = d_max_km * 1000.0

        for i, az in enumerate(az_list, start=1):
            dists, lats, lons = build_profile(lat, lon, az, paso_m)
            elev = get_elevations(lats, lons)

            dh, h10, h90 = compute_delta_h(
                dists,
                elev,
                metodo_dh,
                d_min_custom=d_min_custom_m,
                d_max_custom=d_max_custom_m
            )

            if metodo_dh == "Personalizado (km)":
                rango_txt = f"{d_min_km:.2f}–{d_max_km:.2f}"
            else:
                rango_txt = RANGO_METODO.get(metodo_dh, "")

            results.append({
                "Azimut (°)": az,
                "Δh (m)": dh,
                "h10 (P10, m)": h10,
                "h90 (P90, m)": h90,
                "Método Δh": metodo_dh,
                "Rango (km)": rango_txt
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
# CONTORNO FCC (F(50,50))
# ------------------------------------------------------------
elif categoria == "Contorno FCC":

    st.subheader("📡 Contorno FCC F(50,50) – FM")

    erp_kw = st.number_input("ERP (kW)", value=1.0, min_value=0.01)
    haat_m = st.number_input("HAAT (m)", value=100.0)
    campo_db = st.number_input("Nivel de campo (dBµV/m)", value=54.0)

    curvas = cargar_curvas_fcc()

    if st.button("Calcular contorno FCC"):
        d_km = distancia_fcc_f5050(erp_kw, haat_m, campo_db, curvas)

        st.success(f"📏 Distancia del contorno: **{d_km:.3f} km**")

        azs = np.arange(0, 360, 5)
        pts = []

        for az in azs:
            la, lo = destination_point(lat, lon, az, d_km * 1000)
            pts.append([la, lo])

        m = folium.Map(location=[lat, lon], zoom_start=7)
        folium.Marker([lat, lon], tooltip="Transmisor").add_to(m)
        folium.Polygon(pts, color="blue", fill=True, fill_opacity=0.3).add_to(m)

        st_folium(m, height=550)

def distancia_fcc_f5050(erp_kw, haat_m, campo_db, df):
    """
    Calcula distancia FCC F(50,50) por interpolación real.
    ERP en kW
    HAAT en metros
    Campo en dBµV/m
    """

    # Ajuste ERP (FCC trabaja con 1 kW base)
    campo_1kw = campo_db + 10 * math.log10(erp_kw)

    # Filtrar HAATs disponibles
    haats = np.sort(df["haat_m"].unique())

    if haat_m <= haats.min():
        h1 = h2 = haats.min()
    elif haat_m >= haats.max():
        h1 = h2 = haats.max()
    else:
        h1 = haats[haats <= haat_m].max()
        h2 = haats[haats >= haat_m].min()

    def interp_dist(h):
        sub = df[df["haat_m"] == h].sort_values("field_dbu_1kw")
        return np.interp(
            campo_1kw,
            sub["field_dbu_1kw"][::-1],
            sub["distance_km"][::-1]
        )

    d1 = interp_dist(h1)
    d2 = interp_dist(h2)

    if h1 == h2:
        return d1

    return d1 + (d2 - d1) * (haat_m - h1) / (h2 - h1)

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

    st.subheader("Resultados de Rugosidad Δh")
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
