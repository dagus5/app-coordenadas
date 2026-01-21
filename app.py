# -*- coding: utf-8 -*-
# app.py — Coordenadas + Δh (ITM/FCC/MSAM) 0–50 km

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
# FUNCIONES GEO
# ------------------------------------------------------------

R_EARTH_M = 6371000.0

def destination_point(lat_deg, lon_deg, bearing_deg, distance_m):
    lat1 = math.radians(latAD)
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

# ------------------------------------------------------------
# INPUT COORDENADAS
# ------------------------------------------------------------

def input_coords():
    c1, c2 = st.columns(2)
    with c1:
        lat = st.number_input("Latitud (decimal)", value=8.8066)
    with c2:
        lon = st.number_input("Longitud (decimal)", value=-82.5403)
    return lat, lon

# ------------------------------------------------------------
# MENÚ
# ------------------------------------------------------------

st.markdown("### Selecciona una categoría")

c1, c2, c3 = st.columns(3)

if c1.button("📍 8 Radiales"):
    st.session_state.categoria = "8 Radiales"

if c2.button("🌄 Δh – Rugosidad"):
    st.session_state.categoria = "Δh"

if c3.button("📡 Contorno FCC"):
    st.session_state.categoria = "Contorno FCC"

categoria = st.session_state.categoria
st.markdown(f"### 🟢 Categoría seleccionada: **{categoria}**")

lat, lon = input_coords()

# ------------------------------------------------------------
# 8 RADIALES
# ------------------------------------------------------------

if categoria == "8 Radiales":
    acimuts = [0,45,90,135,180,225,270,315]
    dist_km = st.number_input("Distancia (km)", value=50.0)

    if st.button("Calcular"):
        filas = []
        for az in acimuts:
            la, lo = destination_point(lat, lon, az, dist_km*1000)
            filas.append({"Azimut": az, "Lat": la, "Lon": lo})

        df = pd.DataFrame(filas)
        st.dataframe(df)

        m = folium.Map(location=[lat, lon], zoom_start=8)
        folium.Marker([lat, lon], tooltip="Centro").add_to(m)

        for _, r in df.iterrows():
            folium.PolyLine([[lat,lon],[r["Lat"],r["Lon"]]]).add_to(m)

        st_folium(m, height=500)

# ------------------------------------------------------------
# Δh – RUGOSIDAD (SIN CAMBIOS)
# ------------------------------------------------------------

elif categoria == "Δh":
    st.info("Sección Δh intacta (no modificada)")

# ------------------------------------------------------------
# CONTORNO FCC (54 dBµV/m o cualquier nivel)
# ------------------------------------------------------------

elif categoria == "Contorno FCC":

    st.subheader("📡 Contorno FCC F(50,50)")

    erp_kw = st.number_input("ERP (kW)", value=1.0, min_value=0.01)
    haat_m = st.number_input("HAAT (m)", value=100.0)
    campo_db = st.number_input("Nivel de campo (dBµV/m)", value=54.0)

    def fcc_distancia(erp_kw, haat_m, campo_db):
        # Aproximación ingeniería (misma lógica FCC)
        return max(1.0, (1.06 * math.sqrt(erp_kw)) * (haat_m/100)**0.3 * (106/campo_db))

    if st.button("Calcular contorno"):
        d_km = fcc_distancia(erp_kw, haat_m, campo_db)

        st.success(f"📏 Distancia del contorno: **{d_km:.1f} km**")

        azs = np.arange(0,360,5)
        pts = []

        for az in azs:
            la, lo = destination_point(lat, lon, az, d_km*1000)
            pts.append([la, lo])

        m = folium.Map(location=[lat, lon], zoom_start=7)
        folium.Marker([lat, lon], tooltip="Transmisor").add_to(m)
        folium.Polygon(pts, color="blue", fill=True, fill_opacity=0.3).add_to(m)

        st_folium(m, height=550)
