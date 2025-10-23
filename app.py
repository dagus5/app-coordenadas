import streamlit as st 
import pandas as pd
from pygeodesy.ellipsoidalVincenty import LatLon
import folium
from streamlit_folium import st_folium

# ---------------- CONFIGURACIÓN GENERAL ---------------- #
st.set_page_config(page_title="Calculadora de Coordenadas", layout="wide")

st.title("🧭 Calculadora de Coordenadas")
st.markdown("Selecciona el tipo de cálculo que deseas realizar.")

# Mantener resultados
if "categoria" not in st.session_state:
    st.session_state.categoria = None

# ---------------- FUNCIONES ---------------- #

def mostrar_mapa(df, lat=None, lon=None, categoria=None):
    """Genera el mapa con marcadores y líneas según la categoría"""
    mapa = folium.Map(location=[lat, lon], zoom_start=9)

    if categoria in ["Calculo - 8 Radiales", "Calculo por Azimut"]:
        folium.Marker([lat, lon], tooltip="Punto inicial", icon=folium.Icon(color="red")).add_to(mapa)
        for _, row in df.iterrows():
            lat_fin = float(row["Latitud Final (Decimal)"])
            lon_fin = float(row["Longitud Final (Decimal)"])
            distancia = row.get("Distancia (km)", "")
            acimut = row.get("Acimut (°)", "")
            folium.Marker([lat_fin, lon_fin],
                          tooltip=f"Distancia: {distancia} km | Acimut: {acimut}°",
                          icon=folium.Icon(color="blue", icon="flag")).add_to(mapa)
            folium.PolyLine([[lat, lon], [lat_fin, lon_fin]], color="orange", weight=2).add_to(mapa)

    elif categoria == "Calculo de distancia":
        for _, row in df.iterrows():
            lat1, lon1 = float(row["Latitud 1"]), float(row["Longitud 1"])
            lat2, lon2 = float(row["Latitud 2"]), float(row["Longitud 2"])
            folium.Marker([lat1, lon1], tooltip="Punto 1", icon=folium.Icon(color="red")).add_to(mapa)
            folium.Marker([lat2, lon2], tooltip="Punto 2", icon=folium.Icon(color="blue")).add_to(mapa)
            folium.PolyLine([[lat1, lon1], [lat2, lon2]], color="blue", weight=2).add_to(mapa)

    elif categoria == "Calculo de distancia central":
        for _, row in df.iterrows():
            lat_c = float(row["Latitud central"])
            lon_c = float(row["Longitud central"])
            lat_p = float(row["Latitud punto"])
            lon_p = float(row["Longitud punto"])
            folium.Marker([lat_c, lon_c], tooltip="Central", icon=folium.Icon(color="red")).add_to(mapa)
            folium.Marker([lat_p, lon_p], tooltip="Punto", icon=folium.Icon(color="blue")).add_to(mapa)
            folium.PolyLine([[lat_c, lon_c], [lat_p, lon_p]], color="green", weight=2).add_to(mapa)

    st_folium(mapa, width=700, height=500)

# ---------------- MOSAICOS DE OPCIONES ---------------- #

st.markdown("### 🔹 Selecciona una categoría de cálculo:")

col1, col2 = st.columns(2)

with col1:
    if st.button("📐 Calculo - 8 Radiales", use_container_width=True, type="primary"):
        st.session_state.categoria = "Calculo - 8 Radiales"

    if st.button("🧭 Calculo por Azimut", use_container_width=True):
        st.session_state.categoria = "Calculo por Azimut"

with col2:
    if st.button("📏 Calculo de Distancia", use_container_width=True):
        st.session_state.categoria = "Calculo de distancia"

    if st.button("📍 Calculo de Distancia Central", use_container_width=True):
        st.session_state.categoria = "Calculo de distancia central"

st.divider()

# ---------------- INTERFAZ DE CADA CATEGORÍA ---------------- #

if st.session_state.categoria:

    categoria = st.session_state.categoria
    st.subheader(f"🔸 {categoria}")

    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitud inicial (decimal)", value=8.983, step=0.001)
    with col2:
        lon = st.number_input("Longitud inicial (decimal)", value=-79.519, step=0.001)

    # Datos simulados de ejemplo (aquí van tus cálculos reales)
    if categoria == "Calculo - 8 Radiales":
        datos = {
            "Distancia (km)": [10,10,10,10,50,50,50,50],
            "Acimut (°)": [0,45,90,135,180,225,270,315],
            "Latitud Final (Decimal)": [9.083,9.033,8.983,8.933,9.083,9.033,8.933,8.983],
            "Longitud Final (Decimal)": [-79.519,-79.469,-79.419,-79.469,-79.619,-79.569,-79.619,-79.669]
        }
        df = pd.DataFrame(datos)
        st.markdown("#### 📍 Resultados a 10 km")
        st.dataframe(df[df["Distancia (km)"] == 10], use_container_width=True)
        st.markdown("#### 📍 Resultados a 50 km")
        st.dataframe(df[df["Distancia (km)"] == 50], use_container_width=True)
        mostrar_mapa(df, lat, lon, categoria)

    elif categoria == "Calculo por Azimut":
        datos = {
            "Distancia (km)": [20,70],
            "Acimut (°)": [30,200],
            "Latitud Final (Decimal)": [9.05,8.80],
            "Longitud Final (Decimal)": [-79.40,-79.70]
        }
        df = pd.DataFrame(datos)
        st.dataframe(df, use_container_width=True)
        mostrar_mapa(df, lat, lon, categoria)

    elif categoria == "Calculo de distancia":
        datos = {
            "Latitud 1": [8.983],
            "Longitud 1": [-79.519],
            "Latitud 2": [9.083],
            "Longitud 2": [-79.419]
        }
        df = pd.DataFrame(datos)
        st.dataframe(df, use_container_width=True)
        mostrar_mapa(df, lat, lon, categoria)

    elif categoria == "Calculo de distancia central":
        datos = {
            "Latitud central": [8.983],
            "Longitud central": [-79.519],
            "Latitud punto": [9.083,8.883],
            "Longitud punto": [-79.419,-79.619]
        }
        df = pd.DataFrame(datos)
        st.dataframe(df, use_container_width=True)
        mostrar_mapa(df, lat, lon, categoria)
