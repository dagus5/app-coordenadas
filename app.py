import streamlit as st
import pandas as pd
from pygeodesy.ellipsoidalVincenty import LatLon
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Calculadora de Coordenadas", layout="wide")
st.title("🧭 Calculadora de Coordenadas Geográficas")

# ---------- SESIÓN ----------
if "df_resultado" not in st.session_state:
    st.session_state.df_resultado = None

# ---------- FUNCIONES ----------
def decimal_a_gms(grados_decimales, tipo):
    direccion = {"lat": "N" if grados_decimales >= 0 else "S", "lon": "E" if grados_decimales >= 0 else "W"}[tipo]
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

def calcular_distancia_entre(lat1, lon1, lat2, lon2):
    p1, p2 = LatLon(lat1, lon1), LatLon(lat2, lon2)
    distancia = p1.distanceTo(p2) / 1000
    azimut_ida = p1.initialBearingTo(p2)
    azimut_vuelta = p2.initialBearingTo(p1)
    return distancia, azimut_ida, azimut_vuelta

def mostrar_mapa(df, lat, lon, categoria):
    mapa = folium.Map(location=[lat, lon], zoom_start=9)

    if categoria in ["Calculo - 8 Radiales", "Calculo por Azimut"]:
        # Punto central
        folium.Marker([lat, lon], tooltip="Punto inicial", icon=folium.Icon(color="red", icon="home")).add_to(mapa)
        # Puntos finales + líneas
        for _, row in df.iterrows():
            lat_fin = float(row["Latitud Final (Decimal)"])
            lon_fin = float(row["Longitud Final (Decimal)"])
            distancia = row.get("Distancia (km)", "")
            acimut = row.get("Acimut (°)", "")
            folium.Marker([lat_fin, lon_fin],
                          tooltip=f"Distancia: {distancia} km | Acimut: {acimut}°",
                          icon=folium.Icon(color="blue", icon="flag")).add_to(mapa)
            folium.PolyLine([[lat, lon], [lat_fin, lon_fin]], color="orange", weight=2, opacity=0.8).add_to(mapa)

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

# ---------- INTERFAZ PRINCIPAL ----------
opcion = st.radio(
    "Selecciona una categoría de cálculo:",
    ["Calculo - 8 Radiales", "Calculo por Azimut", "Calculo de distancia", "Calculo de distancia central"],
    horizontal=True
)

# ---------- 1. CALCULO - 8 RADIALES ----------
if opcion == "Calculo - 8 Radiales":
    st.markdown("**Cálculo de 8 puntos a 10 km y 50 km (0°–315°)**")
    col1, col2 = st.columns(2)
    with col1:
        lat_input = st.text_input("Latitud inicial (decimal)", value="8.8066")
    with col2:
        lon_input = st.text_input("Longitud inicial (decimal)", value="-82.5403")

    try:
        lat, lon = float(lat_input), float(lon_input)
    except ValueError:
        st.error("⚠️ Ingrese valores válidos de coordenadas.")
        st.stop()

    if st.button("Calcular 8 radiales"):
        acimuts = [0, 45, 90, 135, 180, 225, 270, 315]
        distancias = [10000, 50000]
        df = calcular_puntos(lat, lon, acimuts, distancias)
        st.session_state.df_resultado = df
        st.success("✅ Cálculo completado.")

# ---------- 2. CALCULO POR AZIMUT ----------
elif opcion == "Calculo por Azimut":
    st.markdown("**Cálculo con azimuts y distancias personalizadas**")
    col1, col2 = st.columns(2)
    with col1:
        lat_input = st.text_input("Latitud inicial (decimal)", value="8.8066")
    with col2:
        lon_input = st.text_input("Longitud inicial (decimal)", value="-82.5403")

    acimuts_str = st.text_input("Ingresa los acimuts separados por coma (ej. 0, 90, 180, 270)", "0,45,90,135,180,225,270,315")
    dist_1 = st.number_input("Primera distancia (km)", value=10)
    dist_2 = st.number_input("Segunda distancia (km)", value=50)

    try:
        lat, lon = float(lat_input), float(lon_input)
        acimuts = [float(a.strip()) for a in acimuts_str.split(",")]
    except ValueError:
        st.error("⚠️ Verifique los datos.")
        st.stop()

    if st.button("Calcular por azimut"):
        distancias = [dist_1 * 1000, dist_2 * 1000]
        df = calcular_puntos(lat, lon, acimuts, distancias)
        st.session_state.df_resultado = df
        st.success("✅ Cálculo completado.")

# ---------- 3. CALCULO DE DISTANCIA ----------
elif opcion == "Calculo de distancia":
    st.markdown("**Distancia y acimut entre dos coordenadas**")
    c1, c2 = st.columns(2)
    with c1:
        lat1 = st.text_input("Latitud 1", value="8.8066")
        lon1 = st.text_input("Longitud 1", value="-82.5403")
    with c2:
        lat2 = st.text_input("Latitud 2", value="8.5000")
        lon2 = st.text_input("Longitud 2", value="-82.3000")

    try:
        lat1, lon1, lat2, lon2 = map(float, [lat1, lon1, lat2, lon2])
    except ValueError:
        st.error("⚠️ Verifique las coordenadas.")
        st.stop()

    if st.button("Calcular distancia"):
        distancia, az_ida, az_vuelta = calcular_distancia_entre(lat1, lon1, lat2, lon2)
        df = pd.DataFrame([{
            "Latitud 1": lat1, "Longitud 1": lon1,
            "Latitud 2": lat2, "Longitud 2": lon2,
            "Distancia (km)": distancia,
            "Azimut ida (°)": az_ida,
            "Azimut vuelta (°)": az_vuelta
        }])
        st.session_state.df_resultado = df
        st.success("✅ Cálculo completado.")

# ---------- 4. CALCULO DE DISTANCIA CENTRAL ----------
elif opcion == "Calculo de distancia central":
    st.markdown("**Desde una coordenada central, calcular distancia y azimut hacia varios puntos**")
    col1, col2 = st.columns(2)
    with col1:
        lat_c = st.text_input("Latitud central", value="8.8066")
    with col2:
        lon_c = st.text_input("Longitud central", value="-82.5403")

    puntos_texto = st.text_area("Coordenadas adicionales (una por línea, formato: lat, lon)", "8.7000, -82.5000\n8.6000, -82.4000")
    try:
        lat_c, lon_c = float(lat_c), float(lon_c)
    except ValueError:
        st.error("⚠️ Coordenadas inválidas.")
        st.stop()

    if st.button("Calcular distancias centrales"):
        filas = []
        for linea in puntos_texto.strip().split("\n"):
            if "," not in linea:
                continue
            lat_p, lon_p = [float(x.strip()) for x in linea.split(",")]
            d, az_ida, az_vuelta = calcular_distancia_entre(lat_c, lon_c, lat_p, lon_p)
            filas.append({
                "Latitud central": lat_c, "Longitud central": lon_c,
                "Latitud punto": lat_p, "Longitud punto": lon_p,
                "Distancia (km)": d, "Azimut ida (°)": az_ida, "Azimut vuelta (°)": az_vuelta
            })
        df = pd.DataFrame(filas)
        st.session_state.df_resultado = df
        st.success("✅ Cálculo completado.")

# ---------- MOSTRAR RESULTADOS ----------
if st.session_state.df_resultado is not None:
    df = st.session_state.df_resultado

    if "Distancia (km)" in df.columns:
        for d in df["Distancia (km)"].unique():
            st.subheader(f"Resultados a {d} km")
            st.dataframe(df[df["Distancia (km)"] == d], use_container_width=True)
    else:
        st.dataframe(df, use_container_width=True)

    mostrar_mapa(df, lat if 'lat' in locals() else lat1, lon if 'lon' in locals() else lon1, opcion)

    csv_data = df.to_csv(index=False, sep=';', encoding='utf-8')
    st.download_button("📥 Descargar resultados en CSV", data=csv_data, file_name="coordenadas_resultado.csv", mime="text/csv")

