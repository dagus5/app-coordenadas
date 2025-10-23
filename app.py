import streamlit as st
import math
import folium
from streamlit_folium import st_folium

# Constantes
R_Tierra = 6371  # Radio de la Tierra en km

def grados_a_radianes(grados):
    return grados * math.pi / 180

def radianes_a_grados(radianes):
    return radianes * 180 / math.pi

def calcular_nueva_coordenada(lat, lon, azimut, distancia_km):
    lat_rad = grados_a_radianes(lat)
    lon_rad = grados_a_radianes(lon)
    az_rad = grados_a_radianes(azimut)
    d_div_r = distancia_km / R_Tierra

    lat2 = math.asin(math.sin(lat_rad)*math.cos(d_div_r) + math.cos(lat_rad)*math.sin(d_div_r)*math.cos(az_rad))
    lon2 = lon_rad + math.atan2(math.sin(az_rad)*math.sin(d_div_r)*math.cos(lat_rad),
                               math.cos(d_div_r)-math.sin(lat_rad)*math.sin(lat2))

    return radianes_a_grados(lat2), radianes_a_grados(lon2)

def calcular_distancia_azimut(lat1, lon1, lat2, lon2):
    lat1_rad = grados_a_radianes(lat1)
    lat2_rad = grados_a_radianes(lat2)
    dlon_rad = grados_a_radianes(lon2 - lon1)

    dlat = lat2_rad - lat1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad)*math.cos(lat2_rad)*math.sin(dlon_rad/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distancia_km = R_Tierra * c

    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = math.cos(lat1_rad)*math.sin(lat2_rad) - math.sin(lat1_rad)*math.cos(lat2_rad)*math.cos(dlon_rad)
    azimut_1a2 = (radianes_a_grados(math.atan2(y, x)) + 360) % 360

    y_rev = math.sin(-dlon_rad) * math.cos(lat1_rad)
    x_rev = math.cos(lat2_rad)*math.sin(lat1_rad) - math.sin(lat2_rad)*math.cos(lat1_rad)*math.cos(-dlon_rad)
    azimut_2a1 = (radianes_a_grados(math.atan2(y_rev, x_rev)) + 360) % 360

    return distancia_km, distancia_km*1000, azimut_1a2, azimut_2a1

def mostrar_mapa(puntos, zoom=8):
    # puntos = list de tuplas (lat, lon, etiqueta)
    if not puntos:
        st.info("No hay puntos para mostrar en el mapa.")
        return
    lat_prom = sum(p[0] for p in puntos) / len(puntos)
    lon_prom = sum(p[1] for p in puntos) / len(puntos)
    m = folium.Map(location=[lat_prom, lon_prom], zoom_start=zoom)
    for lat, lon, etiqueta in puntos:
        folium.Marker([lat, lon], tooltip=etiqueta).add_to(m)
    st_folium(m, width=700, height=450)

# --- Interfaz principal ---

st.title("Calculadora Geodésica Avanzada")

# Inicializar selección en session_state para persistencia
if 'seleccion' not in st.session_state:
    st.session_state.seleccion = None

# Mosaicos (botones) para seleccionar categoría
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("Cálculo de los 8 Radiales"):
        st.session_state.seleccion = "radiales"
with col2:
    if st.button("Cálculo por ángulo"):
        st.session_state.seleccion = "por_angulo"
with col3:
    if st.button("Cálculo de distancia"):
        st.session_state.seleccion = "distancia"
with col4:
    if st.button("Cálculo de distancia central"):
        st.session_state.seleccion = "distancia_central"

if st.session_state.seleccion is None:
    st.info("Selecciona una categoría usando los botones de arriba.")
    st.stop()

# --- Categoría 1: Cálculo de los 8 Radiales ---
if st.session_state.seleccion == "radiales":
    st.header("Cálculo de los 8 Radiales (a 10 km)")
    lat = st.number_input("Latitud inicial (grados decimales)", value=0.0, format="%.6f", key="radiales_lat")
    lon = st.number_input("Longitud inicial (grados decimales)", value=0.0, format="%.6f", key="radiales_lon")
    distancia_km = 10
    azimuts = [0, 45, 90, 135, 180, 225, 270, 315]

    if st.button("Calcular radiales", key="btn_radiales"):
        resultados = []
        for az in azimuts:
            lat_n, lon_n = calcular_nueva_coordenada(lat, lon, az, distancia_km)
            resultados.append((az, lat_n, lon_n))
        st.session_state.resultados_radiales = resultados
        st.session_state.lat_radiales = lat
        st.session_state.lon_radiales = lon

    if 'resultados_radiales' in st.session_state:
        st.success(f"Coordenadas a {distancia_km} km desde punto inicial:")
        for az, lat_n, lon_n in st.session_state.resultados_radiales:
            st.write(f"Azimut {az}° → Lat: {lat_n:.6f}°, Lon: {lon_n:.6f}°")

        puntos_mapa = [(st.session_state.lat_radiales, st.session_state.lon_radiales, "Central")] + \
                      [(az_n[1], az_n[2], f"Az {az_n[0]}°") for az_n in st.session_state.resultados_radiales]
        mostrar_mapa(puntos_mapa)

# --- Categoría 2: Cálculo por ángulo ---
elif st.session_state.seleccion == "por_angulo":
    st.header("Cálculo por ángulo con azimuts y distancias personalizadas")
    lat = st.number_input("Latitud inicial (grados decimales)", value=0.0, format="%.6f", key="angulo_lat")
    lon = st.number_input("Longitud inicial (grados decimales)", value=0.0, format="%.6f", key="angulo_lon")

    dist_10 = st.number_input("Distancia para radial corto (km)", value=10.0, min_value=0.1, format="%.3f", key="dist_corto")
    dist_50 = st.number_input("Distancia para radial largo (km)", value=50.0, min_value=0.1, format="%.3f", key="dist_largo")

    azimuts_str = st.text_area("Introduce azimuts separados por comas (ejemplo: 0,45,90,135)", height=100, key="azimuts_text")

    if st.button("Calcular coordenadas", key="btn_angulo"):
        try:
            azimuts = [float(a.strip()) for a in azimuts_str.split(",") if a.strip() != ""]
            resultados = []
            for az in azimuts:
                lat_corto, lon_corto = calcular_nueva_coordenada(lat, lon, az, dist_10)
                lat_largo, lon_largo = calcular_nueva_coordenada(lat, lon, az, dist_50)
                resultados.append((az, lat_corto, lon_corto, lat_largo, lon_largo))
            st.session_state.resultados_angulo = resultados
            st.session_state.lat_angulo = lat
            st.session_state.lon_angulo = lon
            st.session_state.dist_corto = dist_10
            st.session_state.dist_largo = dist_50
        except Exception as e:
            st.error(f"Error en la entrada de azimuts: {e}")

    if 'resultados_angulo' in st.session_state:
        st.success(f"Coordenadas calculadas desde punto inicial:")
        for az, lat_c, lon_c, lat_l, lon_l in st.session_state.resultados_angulo:
            st.write(f"Azimut {az:.2f}° → {st.session_state.dist_corto} km: Lat {lat_c:.6f}°, Lon {lon_c:.6f}° | "
                     f"{st.session_state.dist_largo} km: Lat {lat_l:.6f}°, Lon {lon_l:.6f}°")

        puntos_mapa = [(st.session_state.lat_angulo, st.session_state.lon_angulo, "Central")]
        for az, lat_c, lon_c, lat_l, lon_l in st.session_state.resultados_angulo:
            puntos_mapa.append((lat_c, lon_c, f"Az {az}° {st.session_state.dist_corto}km"))
            puntos_mapa.append((lat_l, lon_l, f"Az {az}° {st.session_state.dist_largo}km"))
        mostrar_mapa(puntos_mapa, zoom=7)

# --- Categoría 3: Cálculo de distancia ---
elif st.session_state.seleccion == "distancia":
    st.header("Cálculo de distancia y azimut entre dos coordenadas")
    lat1 = st.number_input("Latitud punto 1 (grados decimales)", value=0.0, format="%.6f", key="dist_lat1")
    lon1 = st.number_input("Longitud punto 1 (grados decimales)", value=0.0, format="%.6f", key="dist_lon1")
    lat2 = st.number_input("Latitud punto 2 (grados decimales)", value=0.0, format="%.6f", key="dist_lat2")
    lon2 = st.number_input("Longitud punto 2 (grados decimales)", value=0.0, format="%.6f", key="dist_lon2")

    if st.button("Calcular distancia y azimut", key="btn_distancia"):
        distancia_km, distancia_m, azimut_1a2, azimut_2a1 = calcular_distancia_azimut(lat1, lon1, lat2, lon2)
        st.session_state.resultado_distancia = (distancia_km, distancia_m, azimut_1a2, azimut_2a1)
        st.session_state.puntos_distancia = [(lat1, lon1, "Punto 1"), (lat2, lon2, "Punto 2")]

    if 'resultado_distancia' in st.session_state:
        distancia_km, distancia_m, azimut_1a2, azimut_2a1 = st.session_state.resultado_distancia
        st.success("Resultados:")
        st.write(f"Distancia: {distancia_km:.3f} km ({distancia_m:.1f} metros)")
        st.write(f"Azimut de punto 1 a punto 2: {azimut_1a2:.2f}°")
        st.write(f"Azimut de punto 2 a punto 1: {azimut_2a1:.2f}°")

        mostrar_mapa(st.session_state.puntos_distancia, zoom=7)

# --- Categoría 4: Cálculo de distancia central ---
elif st.session_state.seleccion == "distancia_central":
    st.header("Cálculo de distancia y azimut desde coordenada central a múltiples puntos")

    lat_central = st.number_input("Latitud central (grados decimales)", value=0.0, format="%.6f", key="central_lat")
    lon_central = st.number_input("Longitud central (grados decimales)", value=0.0, format="%.6f", key="central_lon")

    if 'coords_adicionales' not in st.session_state:
        st.session_state.coords_adicionales = []

    col_add, col_clear = st.columns([1,1])
    with col_add:
        if st.button("Agregar coordenada"):
            st.session_state.coords_adicionales.append({"lat": 0.0, "lon": 0.0})
    with col_clear:
        if st.button("Limpiar coordenadas"):
            st.session_state.coords_adicionales = []
            if 'resultados_central' in st.session_state:
                del st.session_state['resultados_central']

    # Mostrar inputs para cada coordenada adicional
    for i, coord in enumerate(st.session_state.coords_adicionales):
        lat_i = st.number_input(f"Latitud punto {i+1}", value=coord["lat"], format="%.6f", key=f"central_lat_{i}")
        lon_i = st.number_input(f"Longitud punto {i+1}", value=coord["lon"], format="%.6f", key=f"central_lon_{i}")
        st.session_state.coords_adicionales[i]["lat"] = lat_i
        st.session_state.coords_adicionales[i]["lon"] = lon_i

    if st.button("Calcular distancias y azimuts", key="btn_central"):
        if not st.session_state.coords_adicionales:
            st.warning("Agrega al menos una coordenada adicional.")
        else:
            resultados = []
            for i, coord in enumerate(st.session_state.coords_adicionales):
                d_km, d_m, az_1a2, az_2a1 = calcular_distancia_azimut(lat_central, lon_central, coord["lat"], coord["lon"])
                resultados.append((i+1, coord["lat"], coord["lon"], d_km, d_m, az_1a2, az_2a1))
            st.session_state.resultados_central = resultados
            st.session_state.lat_central = lat_central
            st.session_state.lon_central = lon_central

    if 'resultados_central' in st.session_state:
        st.success(f"Resultados desde punto central ({st.session_state.lat_central:.6f}, {st.session_state.lon_central:.6f}):")
        for i, lat_i, lon_i, d_km, d_m, az_1a2, az_2a1 in st.session_state.resultados_central:
            st.write(f"Punto {i}: Lat {lat_i:.6f}, Lon {lon_i:.6f} → Distancia: {d_km:.3f} km ({d_m:.1f} m), "
                     f"Azimut ida: {az_1a2:.2f}°, Azimut vuelta: {az_2a1:.2f}°")

        puntos_mapa = [(st.session_state.lat_central, st.session_state.lon_central, "Central")] + \
                      [(c["lat"], c["lon"], f"Punto {i+1}") for i, c in enumerate(st.session_state.coords_adicionales)]
        mostrar_mapa(puntos_mapa, zoom=6)

