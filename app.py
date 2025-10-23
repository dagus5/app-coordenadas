import streamlit as st 
import pandas as pd
from pygeodesy.ellipsoidalVincenty import LatLon
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Calculadora de Coordenadas", layout="wide")

st.title("🧭 Calculadora de Coordenadas a 10 km y 50 km")
st.markdown("Ingresa las coordenadas iniciales y selecciona el tipo de cálculo que deseas realizar.")

# ---------------- FUNCIONES ---------------- #

def mostrar_mapa(df, lat=None, lon=None, categoria=None):
    """Genera el mapa con marcadores y líneas según la categoría"""
    if lat is None or lon is None:
        # Detección automática de coordenadas iniciales
        posibles = [
            ("Latitud central", "Longitud central"),
            ("Latitud 1", "Longitud 1"),
            ("Latitud punto", "Longitud punto")
        ]
        for c1, c2 in posibles:
            if c1 in df.columns and c2 in df.columns:
                lat = float(df[c1].iloc[0])
                lon = float(df[c2].iloc[0])
                break
        else:
            st.warning("⚠️ No se encontraron coordenadas iniciales para mostrar el mapa.")
            return

    mapa = folium.Map(location=[lat, lon], zoom_start=9)

    # --- Cálculo - 8 Radiales o Azimut ---
    if categoria in ["Calculo - 8 Radiales", "Calculo por Azimut"]:
        folium.Marker(
            [lat, lon],
            tooltip="Punto Inicial",
            icon=folium.Icon(color="red", icon="home")
        ).add_to(mapa)

        for _, row in df.iterrows():
            try:
                lat_fin = float(row["Latitud Final (Decimal)"])
                lon_fin = float(row["Longitud Final (Decimal)"])
            except KeyError:
                continue

            distancia = row.get("Distancia (km)", "")
            acimut = row.get("Acimut (°)", "")

            folium.Marker(
                [lat_fin, lon_fin],
                tooltip=f"Distancia: {distancia} km | Acimut: {acimut}°",
                icon=folium.Icon(color="blue", icon="flag")
            ).add_to(mapa)

            folium.PolyLine(
                [[lat, lon], [lat_fin, lon_fin]],
                color="orange",
                weight=2,
                opacity=0.8
            ).add_to(mapa)

    # --- Cálculo de Distancia ---
    elif categoria == "Calculo de distancia":
        for _, row in df.iterrows():
            try:
                lat1, lon1 = float(row["Latitud 1"]), float(row["Longitud 1"])
                lat2, lon2 = float(row["Latitud 2"]), float(row["Longitud 2"])
            except KeyError:
                continue

            folium.Marker([lat1, lon1], tooltip="Punto 1", icon=folium.Icon(color="red")).add_to(mapa)
            folium.Marker([lat2, lon2], tooltip="Punto 2", icon=folium.Icon(color="blue")).add_to(mapa)

            folium.PolyLine(
                [[lat1, lon1], [lat2, lon2]],
                color="blue",
                weight=2,
                opacity=0.7
            ).add_to(mapa)

    # --- Cálculo de Distancia Central ---
    elif categoria == "Calculo de distancia central":
        for _, row in df.iterrows():
            try:
                lat_c = float(row["Latitud central"])
                lon_c = float(row["Longitud central"])
                lat_p = float(row["Latitud punto"])
                lon_p = float(row["Longitud punto"])
            except KeyError:
                continue

            folium.Marker([lat_c, lon_c], tooltip="Central", icon=folium.Icon(color="red")).add_to(mapa)
            folium.Marker([lat_p, lon_p], tooltip="Punto", icon=folium.Icon(color="blue")).add_to(mapa)

            folium.PolyLine(
                [[lat_c, lon_c], [lat_p, lon_p]],
                color="green",
                weight=2,
                opacity=0.7
            ).add_to(mapa)

    # Mostrar mapa
    st_folium(mapa, width=700, height=500)

# ---------------- INTERFAZ ---------------- #

col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitud inicial (decimal)", value=8.983, step=0.001)
with col2:
    lon = st.number_input("Longitud inicial (decimal)", value=-79.519, step=0.001)

opciones = ["Calculo - 8 Radiales", "Calculo por Azimut", "Calculo de distancia", "Calculo de distancia central"]
opcion = st.selectbox("Selecciona el tipo de cálculo", opciones)

# Ejemplo de resultados de prueba
if opcion == "Calculo - 8 Radiales":
    datos = {
        "Distancia (km)": [10, 10, 10, 10, 50, 50, 50, 50],
        "Acimut (°)": [0, 90, 180, 270, 45, 135, 225, 315],
        "Latitud Final (Decimal)": [9.083, 8.983, 8.883, 8.983, 9.033, 8.933, 8.933, 9.033],
        "Longitud Final (Decimal)": [-79.519, -79.419, -79.519, -79.619, -79.469, -79.469, -79.569, -79.569]
    }
    df = pd.DataFrame(datos)

    # --- Mostrar resultados en mosaicos separados ---
    st.subheader("📍 Resultados a 10 km")
    st.dataframe(df[df["Distancia (km)"] == 10])

    st.subheader("📍 Resultados a 50 km")
    st.dataframe(df[df["Distancia (km)"] == 50])

    st.divider()
    st.subheader("🗺️ Mapa de Radiales")
    mostrar_mapa(df, lat, lon, opcion)

elif opcion == "Calculo por Azimut":
    datos = {
        "Distancia (km)": [10, 50],
        "Acimut (°)": [30, 200],
        "Latitud Final (Decimal)": [9.05, 8.80],
        "Longitud Final (Decimal)": [-79.40, -79.70]
    }
    df = pd.DataFrame(datos)

    st.subheader("📍 Resultados del Cálculo por Azimut")
    st.dataframe(df)
    st.divider()
    st.subheader("🗺️ Mapa del Azimut")
    mostrar_mapa(df, lat, lon, opcion)

elif opcion == "Calculo de distancia":
    datos = {
        "Latitud 1": [8.983],
        "Longitud 1": [-79.519],
        "Latitud 2": [9.083],
        "Longitud 2": [-79.419]
    }
    df = pd.DataFrame(datos)

    st.subheader("📍 Cálculo de distancia entre dos puntos")
    st.dataframe(df)
    st.divider()
    mostrar_mapa(df, lat, lon, opcion)

elif opcion == "Calculo de distancia central":
    datos = {
        "Latitud central": [8.983],
        "Longitud central": [-79.519],
        "Latitud punto": [9.083],
        "Longitud punto": [-79.619]
    }
    df = pd.DataFrame(datos)

    st.subheader("📍 Cálculo de distancia central")
    st.dataframe(df)
    st.divider()
    mostrar_mapa(df, lat, lon, opcion)
