# -*- coding: utf-8 -*-
# app.py — Δh ITM / MSAM (PTP) con SRTM / GeoTIFF

import streamlit as st
import numpy as np
import pandas as pd
import math

from pygeodesy.ellipsoidalVincenty import LatLon
import rasterio
from pyproj import Transformer

import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="Δh ITM / MSAM", layout="wide")
st.title("🌄 Cálculo de Δh — ITM / MSAM (PTP)")

R_EARTH = 6371000.0

# ============================================================
# GEODESIA
# ============================================================

def destination_point(lat, lon, az, dist):
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)
    brng = math.radians(az)
    dr = dist / R_EARTH

    lat2 = math.asin(
        math.sin(lat1) * math.cos(dr) +
        math.cos(lat1) * math.sin(dr) * math.cos(brng)
    )

    lon2 = lon1 + math.atan2(
        math.sin(brng) * math.sin(dr) * math.cos(lat1),
        math.cos(dr) - math.sin(lat1) * math.sin(lat2)
    )

    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180


def build_profile(lat, lon, az, step_m):
    d = np.arange(0, 50001, step_m)
    lats, lons = [], []
    for di in d:
        la, lo = destination_point(lat, lon, az, di)
        lats.append(la)
        lons.append(lo)
    return d, lats, lons

# ============================================================
# DEM (SRTM / GeoTIFF)
# ============================================================

class DEM:
    def __init__(self, path):
        self.ds = rasterio.open(path)
        self.transformer = Transformer.from_crs(
            "EPSG:4326", self.ds.crs, always_xy=True
        )

    def elev(self, lat, lon):
        try:
            x, y = self.transformer.transform(lon, lat)
            row, col = self.ds.index(x, y)
            v = self.ds.read(1)[row, col]
            if v == self.ds.nodata:
                return None
            return float(v)
        except:
            return None


def profile_elevations(lats, lons, dem):
    return [dem.elev(la, lo) for la, lo in zip(lats, lons)]

# ============================================================
# Δh ITM / MSAM (OFICIAL)
# ============================================================

def delta_h_itm_msam(dist_m, elev_m):
    """
    Δh = P90 - P10 de los RESIDUOS del perfil
    Perfil 10–50 km, con detrending
    """

    data = [
        (d, h) for d, h in zip(dist_m, elev_m)
        if h is not None and 10000 <= d <= 50000
    ]

    if len(data) < 20:
        return None, None, None

    d = np.array([x[0] for x in data])
    h = np.array([x[1] for x in data])

    # Detrending (paso CLAVE)
    A = np.vstack([d, np.ones(len(d))]).T
    a, b = np.linalg.lstsq(A, h, rcond=None)[0]
    residuals = h - (a * d + b)

    h10 = np.percentile(residuals, 10)
    h90 = np.percentile(residuals, 90)

    return float(h90 - h10), float(h10), float(h90)

# ============================================================
# UI — ENTRADA
# ============================================================

st.subheader("📍 Coordenadas del transmisor")

lat = st.number_input("Latitud (°)", value=8.8066, format="%.6f")
lon = st.number_input("Longitud (°)", value=-82.5403, format="%.6f")

st.subheader("🗺️ Fuente DEM")

dem_option = st.selectbox(
    "Modelo de terreno",
    ["SRTM-1 (30 m)", "SRTM-3 (90 m)", "GeoTIFF personalizado"]
)

dem = None

if dem_option == "SRTM-1 (30 m)":
    dem = DEM("data/SRTM1.tif")
elif dem_option == "SRTM-3 (90 m)":
    dem = DEM("data/SRTM3.tif")
else:
    up = st.file_uploader("Sube DEM GeoTIFF (.tif)", type=["tif"])
    if up:
        with open("temp_dem.tif", "wb") as f:
            f.write(up.read())
        dem = DEM("temp_dem.tif")

st.subheader("⚙️ Parámetros")

az_txt = st.text_input("Azimuts (°)", value="0,45,90,135,180,225,270,315")
step_m = st.number_input("Paso del perfil (m)", value=500, min_value=50, step=50)

# ============================================================
# CÁLCULO
# ============================================================

if st.button("Calcular Δh") and dem:

    az_list = [float(a.strip()) for a in az_txt.split(",") if a.strip()]
    results = []
    profiles = {}

    pb = st.progress(0)
    for i, az in enumerate(az_list):
        dist, lats, lons = build_profile(lat, lon, az, step_m)
        elev = profile_elevations(lats, lons, dem)

        dh, h10, h90 = delta_h_itm_msam(dist, elev)

        results.append({
            "Azimut (°)": az,
            "Δh (m)": dh,
            "h10 (m)": h10,
            "h90 (m)": h90
        })

        profiles[az] = pd.DataFrame({
            "Distancia (km)": dist / 1000,
            "Elevación (m)": elev
        })

        pb.progress((i + 1) / len(az_list))

    df = pd.DataFrame(results).sort_values("Azimut (°)")
    st.success("Cálculo completado")

    # ========================================================
    # RESULTADOS
    # ========================================================

    st.subheader("📊 Resultados Δh")
    st.dataframe(df, use_container_width=True)

    st.markdown(f"**Δh promedio:** {df['Δh (m)'].mean():.2f} m")

    az_sel = st.selectbox("Perfil a visualizar", df["Azimut (°)"])
    prof = profiles[az_sel]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=prof["Distancia (km)"],
        y=prof["Elevación (m)"],
        mode="lines"
    ))
    fig.update_layout(
        title=f"Perfil de terreno — {az_sel}°",
        xaxis_title="Distancia (km)",
        yaxis_title="Elevación (m)"
    )
    st.plotly_chart(fig, use_container_width=True)

    m = folium.Map(location=[lat, lon], zoom_start=8)
    folium.Marker([lat, lon], tooltip="TX").add_to(m)

    for az in az_list:
        pts = []
        for dkm in profiles[az]["Distancia (km)"]:
            la, lo = destination_point(lat, lon, az, dkm * 1000)
            pts.append([la, lo])
        folium.PolyLine(pts).add_to(m)

    st.subheader("🗺️ Radiales")
    st_folium(m, height=520)

    st.download_button(
        "Descargar CSV",
        df.to_csv(index=False).encode("utf-8"),
        "delta_h_msam.csv",
        "text/csv"
    )
