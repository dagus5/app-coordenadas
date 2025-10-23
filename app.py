import streamlit as st
import pandas as pd
from pygeodesy.ellipsoidalVincenty import LatLon
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Calculadora de Coordenadas", layout="wide")
st.title("🧭 Calculadora Avanzada de Coordenadas")

# ---------- SESIÓN PARA MANTENER RESULTADOS ----------
if "df_resultado" not in st.session_state:
    st.session_state.df_resultado = {}
if "categoria" not in st.session_state:
    st.session_state.categoria = "Calculo - 8 Radiales"

# ---------- FUNCIONES ----------
def decimal_a_gms(grados_decimales, tipo):
    direccion = {"lat": "N" if grados_decimales >= 0 else "S",
                 "lon": "E" if grados_decimales >= 0 else "W"}[tipo]
    grados_decimales = abs(grados_decimales)
    grados = int(grados_decimales)
    minutos_decimales = (grados_decimales - grados) * 60
    minutos = int(minutos_decimales)
    segundos = (minutos_decimales - minutos) * 60
    return f"{grados}° {minutos}' {segundos:.8f}\" {direccion}"

def calcular_puntos(lat_inicial, lon_inicial, acimuts, distancias):
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

def calcular_distancia_azimut(lat1, lon1, lat2, lon2):
    punto1 = LatLon(lat1, lon1)
    punto2 = LatLon(lat2, lon2)
    distancia = punto1.distanceTo(punto2)
    acimut_ida = punto1.initialBearingTo(punto2)
    acimut_vuelta = punto2.initialBearingTo(punto1)
    return distancia/1000, acimut_ida, acimut_vuelta

def mostrar_mapa(df, lat, lon, categoria):
    mapa = folium.Map(location=[lat, lon], zoom_start=9)

    if categoria in ["Calculo - 8 Radiales", "Calculo por Azimut"]:
        for _, row in df.iterrows():
            folium.Marker([float(row["Latitud Final (Decimal)"]),
                           float(row["Longitud Final (Decimal)"])],
                          tooltip=f"{row.get('Acimut (°)', '')}° - {row.get('Distancia (km)', '')} km").add_to(mapa)
        folium.Marker([lat, lon], tooltip="Punto inicial", icon=folium.Icon(color="red")).add_to(mapa)

    elif categoria == "Calculo de distancia":
        for _, row in df.iterrows():
            lat2, lon2 = float(row["Latitud 2"]), float(row["Longitud 2"])
            folium.Marker([lat, lon], tooltip="Punto 1", icon=folium.Icon(color="red")).add_to(mapa)
            folium.Marker([lat2, lon2], tooltip="Punto 2", icon=folium.Icon(color="blue")).add_to(mapa)
            folium.PolyLine([[lat, lon], [lat2, lon2]], color="blue", weight=2).add_to(mapa)

    elif categoria == "Calculo de distancia central":
        for _, row in df.iterrows():
            lat_c, lon_c = float(row["Latitud central"]), float(row["Longitud central"])
            lat_p, lon_p = float(row["Latitud punto"]), float(row["Longitud punto"])
            folium.Marker([lat_c, lon_c], tooltip="Central", icon=folium.Icon(color="red")).add_to(mapa)
            folium.Marker([lat_p, lon_p], tooltip="Punto", icon=folium.Icon(color="blue")).add_to(mapa)
            folium.PolyLine([[lat_c, lon_c], [lat_p, lon_p]], color="green", weight=2).add_to(mapa)

    st_folium(mapa, width=700, height=500)

# ---------- BOTONES TIPO MOSAICO ----------
st.markdown("### Selecciona la categoría de cálculo")
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

if col1.button("📍 Calculo - 8 Radiales"):
    st.session_state.categoria = "Calculo - 8 Radiales"
if col2.button("🧭 Calculo por Azimut"):
    st.session_state.categoria = "Calculo por Azimut"
if col3.button("📏 Calculo de distancia"):
    st.session_state.categoria = "Calculo de distancia"
if col4.button("🗺️ Calculo de distancia central"):
    st.session_state.categoria = "Calculo de distancia central"

categoria = st.session_state.categoria
st.markdown(f"### 🟢 Categoría seleccionada: {categoria}")

# ---------- ENTRADA DE COORDENADAS COMÚN ----------
col1, col2 = st.columns(2)
with col1:
    lat_input = st.text_input("Latitud inicial (decimal)", value="8.8066", key=f"{categoria}_lat")
with col2:
    lon_input = st.text_input("Longitud inicial (decimal)", value="-82.5403", key=f"{categoria}_lon")

try:
    lat = float(lat_input)
    lon = float(lon_input)
except ValueError:
    st.error("Por favor ingresa números válidos para latitud y longitud.")
    st.stop()

# ---------- CATEGORÍAS ----------
if categoria == "Calculo - 8 Radiales":
    acimuts = [0, 45, 90, 135, 180, 225, 270, 315]
    distancias = [10000, 50000]
    if st.button("Calcular coordenadas", key="calc_8rad"):
        st.session_state.df_resultado[categoria] = calcular_puntos(lat, lon, acimuts, distancias)
        st.success("✅ Cálculo completado exitosamente.")

elif categoria == "Calculo por Azimut":
    acimuts_input = st.text_input("Ingresa acimuts separados por coma (°)", value="0,45,90,135,180,225,270,315")
    dist10 = st.number_input("Distancia 1 (m)", value=10000)
    dist50 = st.number_input("Distancia 2 (m)", value=50000)
    if st.button("Calcular coordenadas por Azimut", key="calc_azimut"):
        try:
            acimuts = [float(a.strip()) for a in acimuts_input.split(",")]
            distancias = [dist10, dist50]
            st.session_state.df_resultado[categoria] = calcular_puntos(lat, lon, acimuts, distancias)
            st.success("✅ Cálculo completado exitosamente.")
        except:
            st.error("Error en los acimuts ingresados.")

elif categoria == "Calculo de distancia":
    col1, col2 = st.columns(2)
    with col1:
        lat2_input = st.text_input("Latitud 2 (decimal)", value="8.8066", key="lat2")
    with col2:
        lon2_input = st.text_input("Longitud 2 (decimal)", value="-82.5403", key="lon2")
    if st.button("Calcular distancia y acimut", key="calc_dist"):
        try:
            lat2 = float(lat2_input)
            lon2 = float(lon2_input)
            distancia, acimut_ida, acimut_vuelta = calcular_distancia_azimut(lat, lon, lat2, lon2)
            df_result = pd.DataFrame([{
                "Distancia (km)": distancia,
                "Acimut ida (°)": acimut_ida,
                "Acimut vuelta (°)": acimut_vuelta,
                "Latitud 1": lat,
                "Longitud 1": lon,
                "Latitud 2": lat2,
                "Longitud 2": lon2
            }])
            st.session_state.df_resultado[categoria] = df_result
            st.success("✅ Cálculo completado exitosamente.")
        except:
            st.error("Error en las coordenadas ingresadas.")

elif categoria == "Calculo de distancia central":
    num_puntos = st.number_input("Número de coordenadas desde el punto central", value=2, min_value=1)
    puntos = []
    for i in range(num_puntos):
        col1, col2 = st.columns(2)
        with col1:
            lat_p = st.text_input(f"Latitud punto {i+1}", value="8.8066", key=f"lat_central_{i}")
        with col2:
            lon_p = st.text_input(f"Longitud punto {i+1}", value="-82.5403", key=f"lon_central_{i}")
        puntos.append((lat_p, lon_p))
    if st.button("Calcular distancias centrales", key="calc_central"):
        resultados = []
        try:
            for lat_p, lon_p in puntos:
                lat_f = float(lat_p)
                lon_f = float(lon_p)
                distancia, acimut_ida, acimut_vuelta = calcular_distancia_azimut(lat, lon, lat_f, lon_f)
                resultados.append({
                    "Distancia (km)": distancia,
                    "Acimut ida (°)": acimut_ida,
                    "Acimut vuelta (°)": acimut_vuelta,
                    "Latitud central": lat,
                    "Longitud central": lon,
                    "Latitud punto": lat_f,
                    "Longitud punto": lon_f
                })
            st.session_state.df_resultado[categoria] = pd.DataFrame(resultados)
            st.success("✅ Cálculo completado exitosamente.")
        except:
            st.error("Error en las coordenadas ingresadas.")

# ---------- MOSTRAR RESULTADOS Y MAPA ----------
if categoria in st.session_state.df_resultado:
    df = st.session_state.df_resultado[categoria]
    st.subheader("Resultados")

    # Separar por distancia si existe
    if "Distancia (km)" in df.columns and categoria in ["Calculo - 8 Radiales", "Calculo por Azimut"]:
        for d in df["Distancia (km)"].unique():
            st.subheader(f"Resultados a {d} km")
            st.dataframe(df[df["Distancia (km)"] == d], use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

    mostrar_mapa(df, lat, lon, categoria)

    # Descargar CSV
    csv_data = df.to_csv(index=False, sep=';', encoding='utf-8')
    st.download_button("📥 Descargar resultados en CSV", data=csv_data,
                       file_name=f"{categoria.replace(' ', '_')}.csv", mime="text/csv")