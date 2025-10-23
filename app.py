import streamlit as st
import math

# Constantes
R_Tierra = 6371  # Radio de la Tierra en km

def grados_a_radianes(grados):
    return grados * math.pi / 180

def radianes_a_grados(radianes):
    return radianes * 180 / math.pi

def calcular_nueva_coordenada(lat, lon, azimut, distancia_km):
    """
    Calcula nueva coordenada dado punto inicial, azimut (grados) y distancia (km).
    Fórmula de la esfera (simplificada).
    """
    lat_rad = grados_a_radianes(lat)
    lon_rad = grados_a_radianes(lon)
    az_rad = grados_a_radianes(azimut)
    d_div_r = distancia_km / R_Tierra

    lat2 = math.asin(math.sin(lat_rad)*math.cos(d_div_r) + math.cos(lat_rad)*math.sin(d_div_r)*math.cos(az_rad))
    lon2 = lon_rad + math.atan2(math.sin(az_rad)*math.sin(d_div_r)*math.cos(lat_rad),
                               math.cos(d_div_r)-math.sin(lat_rad)*math.sin(lat2))

    return radianes_a_grados(lat2), radianes_a_grados(lon2)

def calcular_distancia_azimut(lat1, lon1, lat2, lon2):
    """
    Calcula distancia (km) y azimut (grados) entre dos coordenadas.
    Azimut de punto1 a punto2 y viceversa.
    """
    lat1_rad = grados_a_radianes(lat1)
    lat2_rad = grados_a_radianes(lat2)
    dlon_rad = grados_a_radianes(lon2 - lon1)

    # Distancia con fórmula del haversine
    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(dlon_rad/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distancia_km = R_Tierra * c

    # Azimut de punto1 a punto2
    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad)*math.sin(lat2_rad) - math.sin(lat1_rad)*math.cos(lat2_rad)*math.cos(dlon_rad)
    azimut_1a2 = (radianes_a_grados(math.atan2(y, x)) + 360) % 360

    # Azimut de punto2 a punto1
    y_rev = math.sin(-dlon_rad) * math.cos(lat1_rad)
    x_rev = math.cos(lat2_rad)*math.sin(lat1_rad) - math.sin(lat2_rad)*math.cos(lat1_rad)*math.cos(-dlon_rad)
    azimut_2a1 = (radianes_a_grados(math.atan2(y_rev, x_rev)) + 360) % 360

    return distancia_km, distancia_km*1000, azimut_1a2, azimut_2a1

# --- Interfaz Streamlit ---

st.title("Calculadora Geodésica Multicategoría")

categoria = st.sidebar.selectbox("Elige tipo de cálculo", [
    "Coordenadas a distancia fija y azimut",
    "Calcular coordenadas desde azimut(es) y punto central",
    "Distancia y azimut entre dos coordenadas"
])

if categoria == "Coordenadas a distancia fija y azimut":
    st.header("Cálculo de coordenadas a 10 km con azimut fijo")
    lat = st.number_input("Latitud inicial (grados decimales)", value=0.0, format="%.6f")
    lon = st.number_input("Longitud inicial (grados decimales)", value=0.0, format="%.6f")
    distancia_km = 10  # fijo 10 km
    azimut = st.number_input("Azimut (grados)", min_value=0.0, max_value=360.0, value=0.0, format="%.2f")

    if st.button("Calcular coordenada"):
        lat_nueva, lon_nueva = calcular_nueva_coordenada(lat, lon, azimut, distancia_km)
        st.success(f"Coordenada a {distancia_km} km y azimut {azimut}°:")
        st.write(f"Latitud: {lat_nueva:.6f}°")
        st.write(f"Longitud: {lon_nueva:.6f}°")

elif categoria == "Calcular coordenadas desde azimut(es) y punto central":
    st.header("Calcular coordenadas para múltiples azimuts desde punto central")
    lat_central = st.number_input("Latitud central (grados decimales)", value=0.0, format="%.6f")
    lon_central = st.number_input("Longitud central (grados decimales)", value=0.0, format="%.6f")
    distancia_km = st.number_input("Distancia (km)", min_value=0.0, value=10.0, format="%.3f")

    azimuts_str = st.text_area("Introduce azimuts separados por comas (ejemplo: 0,45,90,135)", height=100)
    if st.button("Calcular coordenadas"):
        try:
            azimuts = [float(a.strip()) for a in azimuts_str.split(",") if a.strip() != ""]
            resultados = []
            for az in azimuts:
                lat_n, lon_n = calcular_nueva_coordenada(lat_central, lon_central, az, distancia_km)
                resultados.append((az, lat_n, lon_n))
            st.write(f"Coordenadas calculadas a {distancia_km} km desde punto central:")
            for az, lat_n, lon_n in resultados:
                st.write(f"Azimut {az:.2f}° → Lat: {lat_n:.6f}°, Lon: {lon_n:.6f}°")
        except Exception as e:
            st.error(f"Error en la entrada de azimuts: {e}")

elif categoria == "Distancia y azimut entre dos coordenadas":
    st.header("Calcular distancia y azimut entre dos puntos")
    lat1 = st.number_input("Latitud punto 1 (grados decimales)", value=0.0, format="%.6f", key="lat1")
    lon1 = st.number_input("Longitud punto 1 (grados decimales)", value=0.0, format="%.6f", key="lon1")
    lat2 = st.number_input("Latitud punto 2 (grados decimales)", value=0.0, format="%.6f", key="lat2")
    lon2 = st.number_input("Longitud punto 2 (grados decimales)", value=0.0, format="%.6f", key="lon2")

    if st.button("Calcular distancia y azimut"):
        distancia_km, distancia_m, azimut_1a2, azimut_2a1 = calcular_distancia_azimut(lat1, lon1, lat2, lon2)
        st.success("Resultados:")
        st.write(f"Distancia: {distancia_km:.3f} km ({distancia_m:.1f} metros)")
        st.write(f"Azimut de punto 1 a punto 2: {azimut_1a2:.2f}°")
        st.write(f"Azimut de punto 2 a punto 1: {azimut_2a1:.2f}°")
