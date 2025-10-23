import streamlit as st
import pandas as pd
from pygeodesy.ellipsoidalVincenty import LatLon
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Calculadora de Coordenadas", layout="wide")

st.title("🧭 Calculadora de Coordenadas a 10 km y 50 km")
st.markdown("Ingresa las coordenadas iniciales y obtén los puntos finales a diferentes acimuts (0°–315°).")

# Entrada de coordenadas
col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitud inicial (decimal)", value=8.8066)
with col2:
    lon = st.number_input("Longitud inicial (decimal)", value=-82.5403)

# Lista de acimuts
acimuts = [0, 45, 90, 135, 180, 225, 270, 315]
distancias = [10000, 50000]  # en metros

def decimal_a_gms(grados_decimales, tipo):
    direccion = {"lat": "N" if grados_decimales >= 0 else "S", "lon": "E" if grados_decimales >= 0 else "W"}[tipo]
    grados_decimales = abs(grados_decimales)
    grados = int(grados_decimales)
    minutos_decimales = (grados_decimales - grados) * 60
    minutos = int(minutos_decimales)
    segundos = (minutos_decimales - minutos) * 60
    return f"{grados}° {minutos}' {segundos:.2f}\" {direccion}"

def calcular_puntos(lat_inicial, lon_inicial):
    punto_referencia = LatLon(lat_inicial, lon_inicial)
    resultados = []
    for distancia in distancias:
        for acimut in acimuts:
            punto_final = punto_referencia.destination(distancia, acimut)
            resultados.append({
                "Distancia (km)": distancia/1000,
                "Acimut (°)": acimut,
                "Latitud Final (Decimal)": punto_final.lat,
                "Longitud Final (Decimal)": punto_final.lon,
                "Latitud (GMS)": decimal_a_gms(punto_final.lat, "lat"),
                "Longitud (GMS)": decimal_a_gms(punto_final.lon, "lon")
            })
    return pd.DataFrame(resultados)

if st.button("Calcular coordenadas"):
    df = calcular_puntos(lat, lon)
    st.success("✅ Cálculo completado exitosamente.")

    st.dataframe(df)

    # Mapa interactivo
    mapa = folium.Map(location=[lat, lon], zoom_start=9)
    for _, row in df.iterrows():
        folium.Marker([row["Latitud Final (Decimal)"], row["Longitud Final (Decimal)"]],
                      tooltip=f"{row['Acimut (°)']}° - {row['Distancia (km)']} km").add_to(mapa)
    folium.Marker([lat, lon], tooltip="Punto inicial", icon=folium.Icon(color="red")).add_to(mapa)
    st_folium(mapa, width=700, height=500)

    # Descargar Excel
    st.download_button("📥 Descargar resultados en Excel", data=df.to_csv(index=False).encode('utf-8'),
                       file_name="coordenadas_resultado.csv", mime="text/csv")
