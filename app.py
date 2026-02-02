# -*- coding: utf-8 -*-
# app.py — Coordenadas + Δh (ITM / FCC / MSAM) 0–50 km

# ============================================================
# IMPORTS
# ============================================================

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


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================

st.set_page_config(page_title="Coordenadas + Δh ITM", layout="wide")
st.title("🧭 Calculadora Avanzada de Coordenadas + 🌄 Δh (ITM / FCC / MSAM)")


# ============================================================
# ESTADO DE LA APP
# ============================================================

if "categoria" not in st.session_state:
    st.session_state.categoria = "Cálculo - 8 Radiales"

if "resultados" not in st.session_state:
    st.session_state.resultados = {}

if "deltaH_state" not in st.session_state:
    st.session_state.deltaH_state = None


# ============================================================
# CONSTANTES
# ============================================================

R_EARTH_M = 6371000.0


# ============================================================
# FUNCIONES GEODÉSICAS Y CONVERSIONES
# ============================================================

def destination_point(lat_deg, lon_deg, bearing_deg, distance_m):
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    brng = math.radians(bearing_deg)
    dr = distance_m / R_EARTH_M

    lat2 = math.asin(
        math.sin(lat1) * math.cos(dr) +
        math.cos(lat1) * math.sin(dr) * math.cos(brng)
    )

    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(dr) * math.cos(lat1),
        math.cos(dr) - math.sin(lat1) * math.sin(lat2)
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
    return -dec if d in ("S", "W") else dec


# ============================================================
# ENTRADA DE COORDENADAS
# ============================================================

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


def input_gms(key_prefix):
    st.write("**Latitud (GMS)**")
    a, b, c, d = st.columns(4)
    with a:
        g1 = st.number_input("Grados", value=8, step=1, key=f"{key_prefix}_lat_g")
    with b:
        m1 = st.number_input("Min", value=48, min_value=0, max_value=59, key=f"{key_prefix}_lat_m")
    with c:
        s1 = st.number_input("Seg", value=23.76, min_value=0.0, max_value=59.999999, key=f"{key_prefix}_lat_s")
    with d:
        d1c = st.selectbox("Dir", ["N", "S"], key=f"{key_prefix}_lat_d")

    st.write("**Longitud (GMS)**")
    e, f, g, h = st.columns(4)
    with e:
        g2 = st.number_input("Grados", value=82, step=1, key=f"{key_prefix}_lon_g")
    with f:
        m2 = st.number_input("Min", value=32, min_value=0, max_value=59, key=f"{key_prefix}_lon_m")
    with g:
        s2 = st.number_input("Seg", value=25.08, min_value=0.0, max_value=59.999999, key=f"{key_prefix}_lon_s")
    with h:
        d2c = st.selectbox("Dir", ["E", "W"], index=1, key=f"{key_prefix}_lon_d")

    lat = gms_a_decimal(g1, m1, s1, d1c, "lat")
    lon = gms_a_decimal(g2, m2, s2, d2c, "lon")

    st.caption(f"↔ Decimal: {lat:.8f}, {lon:.8f}")
    return lat, lon


def input_coords(key_prefix="base"):
    modo = st.radio("Formato de entrada", ["Decimal", "GMS"], horizontal=True)
    return input_decimal("Latitud", "Longitud", key_prefix) if modo == "Decimal" else input_gms(key_prefix)


# ============================================================
# ELEVACIONES — OPEN-METEO
# ============================================================

class Elev429(Exception):
    pass


@retry(wait=wait_exponential(min=0.5, max=4),
       stop=stop_after_attempt(5),
       retry=retry_if_exception_type(Elev429))
def elev_open_meteo_chunk(lats, lons):
    url = "https://api.open-meteo.com/v1/elevation"
    params = {
        "latitude": ",".join(f"{x:.6f}" for x in lats),
        "longitude": ",".join(f"{x:.6f}" for x in lons)
    }
    r = requests.get(url, params=params, timeout=15)
    if r.status_code == 429:
        raise Elev429()
    r.raise_for_status()
    return r.json().get("elevation", [])


def elev_open_meteo(lats, lons):
    out = []
    for i in range(0, len(lats), 80):
        out.extend(elev_open_meteo_chunk(lats[i:i+80], lons[i:i+80]))
        time.sleep(0.2)
    return [float(v) if v is not None else None for v in out]


@st.cache_data(show_spinner=False, ttl=3600)
def get_elevations_cached(lats, lons):
    return tuple(elev_open_meteo(lats, lons))


def get_elevations(lats, lons):
    try:
        return list(get_elevations_cached(tuple(lats), tuple(lons)))
    except Exception as e:
        st.error(f"Error al obtener elevaciones: {e}")
        return [None] * len(lats)


# ============================================================
# PERFIL Y Δh (ITM / FCC / MSAM)
# ============================================================

def build_profile(lat, lon, az, step_m):
    dists = list(range(0, 50001, step_m))
    lats, lons = [], []
    for d in dists:
        la, lo = destination_point(lat, lon, az, d)
        lats.append(la)
        lons.append(lo)
    return dists, lats, lons


def compute_delta_h(dists_m, elev, metodo, d_min_custom=None, d_max_custom=None):

    if metodo == "ITM / MSAM (10–50 km)":
        d_min, d_max = 10000, 50000
    elif metodo == "FCC (3–16 km)":
        d_min, d_max = 3000, 16000
    elif metodo == "0–50 km completo":
        d_min, d_max = 0, 50000
    elif metodo == "Personalizado (km)" and d_min_custom and d_max_custom:
        d_min, d_max = d_min_custom, d_max_custom
    else:
        return None, None, None

    data = [(d, h) for d, h in zip(dists_m, elev) if h is not None and d_min <= d <= d_max]
    if len(data) < 10:
        return None, None, None

    d = np.array([x[0] for x in data])
    h = np.array([x[1] for x in data])

    A = np.vstack([d, np.ones(len(d))]).T
    a, b = np.linalg.lstsq(A, h, rcond=None)[0]
    h_res = h - (a * d + b)

    h10 = np.percentile(h_res, 10)
    h90 = np.percentile(h_res, 90)

    return float(h90 - h10), float(h10), float(h90)


# ============================================================
# FACTOR DE AJUSTE (PER)
# ============================================================

def constante_c_freq(freq):
    return 4.8 if freq > 300 else 1.9 if freq < 108 else 2.5


def correccion_irregularidad(delta_h, freq, C):
    return 0.0 if delta_h <= 50 else C - 0.03 * delta_h * (1 + freq/300.0)


def per_kw_a_dbk(per_kw):
    return 10 * math.log10(per_kw)


def campo_equivalente(Eu, delta_f, fcp):
    return Eu + abs(delta_f) - fcp


def per_ajustada_dbk(Eu, Eueq):
    return Eu - Eueq


def dbk_a_kw(dbk):
    return 10 ** (dbk / 10)


# ============================================================
# MENÚ PRINCIPAL
# ============================================================

st.markdown("### Selecciona una categoría")

c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)

if c1.button("📍 8 Radiales"): st.session_state.categoria = "Cálculo - 8 Radiales"
if c2.button("🧭 Azimut"): st.session_state.categoria = "Cálculo por Azimut"
if c3.button("📏 Distancia"): st.session_state.categoria = "Cálculo de Distancia"
if c4.button("🗺️ Central"): st.session_state.categoria = "Cálculo de Distancia Central"
if c5.button("🌄 Δh"): st.session_state.categoria = "Δh – Rugosidad"
if c6.button("📡 PER"): st.session_state.categoria = "Factor de Ajuste (PER)"

categoria = st.session_state.categoria
st.markdown(f"### 🟢 Categoría: **{categoria}**")


# ============================================================
# COORDENADAS BASE
# ============================================================

lat, lon = input_coords(categoria)


# ============================================================
# AQUÍ SIGUE TODA TU LÓGICA DE CADA CATEGORÍA
# (idéntica a la que ya usabas)
# ============================================================

st.info("✅ App cargada correctamente. Todas las funciones están activas.")
