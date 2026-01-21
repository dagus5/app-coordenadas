# -*- coding: utf-8 -*-
# app.py — Coordenadas + Δh (ITM/FCC/MSAM) 0–50 km + Contorno FCC
# ------------------------------------------------------------

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

st.set_page_config(page_title="Coordenadas + Δh + FCC", layout="wide")
st.title("🧭 Calculadora Avanzada de Coordenadas + 🌄 Δh + 📡 Contorno FCC")

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
# CONSTANTES
# ------------------------------------------------------------

R_EARTH_M = 6371000.0

# ------------------------------------------------------------
# >>> NUEVO: FCC F(50,50) – MODELO APROXIMADO
# ------------------------------------------------------------

def fcc_field_strength(erp_kw, haat_m, dist_km):
    if dist_km <= 0:
        return None
    erp_dbk = 10 * math.log10(erp_kw)
    loss = 32.45 + 20 * math.log10(dist_km) + 20 * math.log10(100)
    haat_corr = 0.1 * math.sqrt(max(haat_m, 0))
    return 106.92 + erp_dbk + haat_corr - loss


def fcc_distance_for_field(erp_kw, haat_m, target_dbuv):
    for d in np.linspace(1, 150, 1500):
        e = fcc_field_strength(erp_kw, haat_m, d)
        if e is not None and e <= target_dbuv:
            return d
    return None

# ------------------------------------------------------------
# FUNCIONES GEO
# ------------------------------------------------------------

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

lat = st.number_input("Latitud (decimal)", value=8.8066)
lon = st.number_input("Longitud (decimal)", value=-82.5403)

# ------------------------------------------------------------
# >>> NUEVA CATEGORÍA: CONTORNO FCC
# ------------------------------------------------------------

if categoria == "Contorno FCC":

    st.subheader("Parámetros del Contorno FCC")

    erp_kw = st.number_input("ERP (kW)", value=1.0, min_value=0.01)
    haat_m = st.number_input("HAAT (m)", value=100.0, min_value=1.0)
    campo = st.number_input("Campo objetivo (dBµV/m)", value=54.0)

    tipo_antena = st.selectbox(
        "Tipo de antena",
        ["Omnidireccional", "Direccional (CSV)"]
    )

    patron = None
    if tipo_antena == "Direccional (CSV)":
        csv_file = st.file_uploader(
            "Cargar patrón (azimut, atten_db)", type=["csv"]
        )
        if csv_file:
            patron = pd.read_csv(csv_file)

    if st.button("Calcular contorno FCC"):

        azimuts = np.arange(0, 360, 5)
        puntos = []

        for az in azimuts:
            att = 0
            if patron is not None:
                row = patron.iloc[(patron["azimut"] - az).abs().argsort()[:1]]
                att = float(row["atten_db"].values[0])

            dist = fcc_distance_for_field(
                erp_kw * (10 ** (-att / 10)),
                haat_m,
                campo
            )

            if dist:
                la, lo = destination_point(lat, lon, az, dist * 1000)
                puntos.append([la, lo])

        m = folium.Map(location=[lat, lon], zoom_start=8)
        folium.Marker([lat, lon], tooltip="Transmisor",
                      icon=folium.Icon(color="red")).add_to(m)

        folium.Polygon(
            locations=puntos,
            color="blue",
            fill=True,
            fill_opacity=0.35,
            tooltip=f"Contorno {campo} dBµV/m"
        ).add_to(m)

        st_folium(m, width=None, height=550)
