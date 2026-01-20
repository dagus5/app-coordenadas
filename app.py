# -*- coding: utf-8 -*-
# app.py — Coordenadas + Δh + Cobertura FCC

import streamlit as st
import pandas as pd
import numpy as np
import math
import folium
from streamlit_folium import st_folium
from pygeodesy.ellipsoidalVincenty import LatLon
from scipy.interpolate import interp1d

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

st.set_page_config(page_title="Radio – ITM / FCC", layout="wide")
st.title("📡 Herramienta Técnica de Radio – ITM / FCC")

# ------------------------------------------------------------
# FUNCIONES GEOGRÁFICAS
# ------------------------------------------------------------

R_EARTH = 6371000

def destination_point(lat, lon, az, dist_m):
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brng = math.radians(az)
    dr = dist_m / R_EARTH

    lat2 = math.asin(
        math.sin(lat1) * math.cos(dr)
        + math.cos(lat1) * math.sin(dr) * math.cos(brng)
    )

    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(dr) * math.cos(lat1),
        math.cos(dr) - math.sin(lat1) * math.sin(lat2)
    )

    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180

# ------------------------------------------------------------
# FCC – TABLAS Y MODELOS
# ------------------------------------------------------------

@st.cache_data
def cargar_tabla_fcc():
    df = pd.read_excel("contorno 2.xlsx")
    df.columns = ["haat_m", "dist_km"]
    return df.sort_values("haat_m")

df_fcc = cargar_tabla_fcc()

def distancia_grado_b(haat_m):
    f = interp1d(
        df_fcc["haat_m"],
        df_fcc["dist_km"],
        kind="linear",
        fill_value="extrapolate"
    )
    return float(f(haat_m))

def erp_efectiva(erp_kw, gan_db):
    return erp_kw * (10 ** (gan_db / 10))

def distancia_direccional(haat_m, erp_kw):
    base = distancia_grado_b(haat_m)
    return base * (erp_kw ** 0.25)

def leer_patron_csv(file):
    df = pd.read_csv(file)
    df.columns = ["azimut", "ganancia_db"]
    return df.sort_values("azimut")

def contorno_fcc_omni(lat, lon, haat_m, paso=5):
    d_km = distancia_grado_b(haat_m)
    pts = []
    for az in range(0, 360, paso):
        la, lo = destination_point(lat, lon, az, d_km * 1000)
        pts.append([la, lo])
    return pts

def contorno_fcc_direccional(lat, lon, haat_m, erp_kw, patron):
    pts = []
    for _, r in patron.iterrows():
        az = float(r["azimut"])
        g = float(r["ganancia_db"])
        erp_eff = erp_efectiva(erp_kw, g)
        d_km = distancia_direccional(haat_m, erp_eff)
        la, lo = destination_point(lat, lon, az, d_km * 1000)
        pts.append([la, lo])
    return pts

# ------------------------------------------------------------
# INPUT COORDENADAS
# ------------------------------------------------------------

st.sidebar.header("📍 Coordenadas del Transmisor")

lat = st.sidebar.number_input("Latitud (decimal)", value=8.8066, format="%.6f")
lon = st.sidebar.number_input("Longitud (decimal)", value=-82.5403, format="%.6f")

# ------------------------------------------------------------
# MENÚ
# ------------------------------------------------------------

categoria = st.sidebar.radio(
    "Selecciona módulo",
    ["Cobertura FCC – Grado B"]
)

# ------------------------------------------------------------
# COBERTURA FCC
# ------------------------------------------------------------

if categoria == "Cobertura FCC – Grado B":

    st.subheader("Cobertura FCC – Grado B (54 dBµV/m)")

    haat_m = st.number_input("HAAT (m)", value=120.0)
    erp_kw = st.number_input("ERP máxima (kW)", value=10.0)

    tipo_antena = st.selectbox(
        "Tipo de antena",
        ["Omnidireccional", "Direccional"]
    )

    patron_df = None
    if tipo_antena == "Direccional":
        archivo = st.file_uploader(
            "Cargar patrón direccional (CSV)",
            type=["csv"]
        )
        if archivo:
            patron_df = leer_patron_csv(archivo)
            st.dataframe(patron_df)

    if st.button("Calcular cobertura FCC"):
        if tipo_antena == "Omnidireccional":
            contorno = contorno_fcc_omni(lat, lon, haat_m)
        else:
            if patron_df is None:
                st.error("Debes cargar el patrón direccional")
                st.stop()
            contorno = contorno_fcc_direccional(
                lat, lon, haat_m, erp_kw, patron_df
            )

        m = folium.Map(location=[lat, lon], zoom_start=8)

        folium.Marker(
            [lat, lon],
            tooltip="Transmisor FM",
            icon=folium.Icon(color="red", icon="tower")
        ).add_to(m)

        folium.Polygon(
            contorno,
            color="blue",
            fill=True,
            fill_opacity=0.35,
            tooltip="Cobertura FCC – Grado B"
        ).add_to(m)

        st_folium(m, height=550)

        st.success("Cobertura FCC calculada correctamente")

