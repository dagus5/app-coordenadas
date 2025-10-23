import streamlit as st
import pandas as pd
from pygeodesy.ellipsoidalVincenty import LatLon
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Calculadora de Coordenadas", layout="wide")
st.title("🧭 Calculadora de Coordenadas a 10 km y 50 km")
st.markdown("Ingresa las coordenadas iniciales y obtén los puntos finales a diferentes acimuts (0°–315°).")

# ---------- SESIÓN PARA MANTENER RESULTADOS ----------
if "df_resultado" not in st.session_state:
    st.session_state.df_resultado = None

# Entrada de coordenadas como texto para preservar decimales
col1, col2 = st.columns(2)
with col1:
    lat_input = st.text_input("Latitud inicial (decimal)", value="8.8066")
with col2:
    lon_input = st.text_input("Longitud inicial (decimal)", value="-82.5403")

# Convertir a float
try:
    lat = float(lat_input)
    lon = float(lon_input)
except ValueError:
    st.error("Por favor ingresa números válidos para latitud y longitud.")
    st.stop()

# Lista de acimuts y distancias
acimuts = [0, 45, 90, 135, 180, 225, 270, 315]
distancias = [10000, 50000]  # en metros

def decimal_a_gms(grados_decimales, tipo):
    direccion = {"lat": "N" if grados_decimales >= 0 else "S", "lon": "E" if grados_decimales >= 0 else "W"}[tipo]
    grados_decimales = abs(grados_decimales)
    grados = int(grados_decimales)
    minutos_decimales = (grados_decimales - grados) * 60
    minutos = int(minutos_decimales)
    segundos = (minutos_decimales - minutos) * 60
    return f"{grados}° {minutos}' {segundos:.8f}\" {direccion}"  # alta precisión

def calcular_puntos(lat_inicial, lon_inicial):
    punto_referencia = LatLon(lat_inicial, lon_inicial)
    resultados = []
    for distancia in distancias:
        for acimut in acimuts:
            punto_final = punto_referencia.destination(distancia, acimut)
            resultados.append({
                "Distancia (km)": distancia / 1000,
                "Acimut (°)": acimut,
                "Latitud Final (Decimal)": f"{punto_final.lat:.10f}",
                "Longitud Final (Decimal)": f"{punto_final.lon:.10f}",
                "Latitud (GMS)": decimal_a_gms(punto_final.lat, "lat"),
                "Longitud (GMS)": decimal_a_gms(punto_final.lon, "lon")
            })
    return pd.DataFrame(resultados)

# ---------- BOTÓN PARA CALCULAR ----------
if st.button("Calcular coordenadas"):
    st.session_state.df_resultado = calcular_puntos(lat, lon)
    st.success("✅ Cálculo completado exitosamente.")

# ---------- MOSTRAR RESULTADO SI EXISTE ----------
if st.session_state.df_resultado is not None:
    df = st.session_state.df_resultado

    # Separar resultados por distancia
    df_10km = df[df["Distancia (km)"] == 10]
    df_50km = df[df["Distancia (km)"] == 50]

    # Mostrar tabla 10 km
    st.subheader("Resultados a 10 km")
    st.dataframe(df_10km, use_container_width=True)

    # Mostrar tabla 50 km
    st.subheader("Resultados a 50 km")
    st.dataframe(df_50km, use_container_width=True)

    # Mapa interactivo con todos los puntos
    mapa = folium.Map(location=[lat, lon], zoom_start=9)
    for _, row in df.iterrows():
        folium.Marker([float(row["Latitud Final (Decimal)"]), float(row["Longitud Final (Decimal)"])],
                      tooltip=f"{row['Acimut (°)']}° - {row['Distancia (km)']} km").add_to(mapa)
    folium.Marker([lat, lon], tooltip="Punto inicial", icon=folium.Icon(color="red")).add_to(mapa)
    st_folium(mapa, width=700, height=500)

    # Descargar CSV completo con separador punto y coma para Excel
    csv_data = df.to_csv(index=False, sep=';', encoding='utf-8')
    st.download_button("📥 Descargar resultados en CSV", data=csv_data, file_name="coordenadas_resultado.csv", mime="text/csv")

