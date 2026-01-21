# -*- coding: utf-8 -*-
# app.py — Coordenadas + Δh (ITM/FCC/MSAM) 0–50 km
# + Contorno FCC F(50,50)

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

# ------------------------------------------------------------
# MENÚ DE CATEGORÍAS
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

lat = st.number_input("Latitud (decimal)", value=8.8066, format="%.6f")
lon = st.number_input("Longitud (decimal)", value=-82.5403, format="%.6f")

# ------------------------------------------------------------
# CÁLCULOS
# ------------------------------------------------------------

if categoria == "Contorno FCC":
    st.subheader("📡 Contorno FCC F(50,50)")

    erp_kw = st.number_input("ERP (kW)", value=10.0, min_value=0.1)
    haat_m = st.number_input("HAAT (m)", value=150.0, min_value=30.0)
    nivel = st.number_input("Nivel de campo (dBµV/m)", value=54.0)

    if st.button("Calcular Contorno FCC"):
        # Modelo FCC simplificado y estable
        distancia_km = 1.06 * math.sqrt(erp_kw) * (haat_m ** 0.25)

        st.session_state.resultados["Contorno FCC"] = distancia_km

        st.success(f"Distancia del contorno {nivel:.0f} dBµV/m: **{distancia_km:.1f} km**")

        m = folium.Map(location=[la]()

