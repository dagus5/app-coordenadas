
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

# ---------- FUNCIONES DE CONVERSIÓN ----------
def decimal_a_gms(grados_decimales, tipo):
    direccion = {"lat": "N" if grados_decimales >= 0 else "S",
                 "lon": "E" if grados_decimales >= 0 else "W"}[tipo]
    grados_decimales = abs(grados_decimales)
    grados = int(grados_decimales)
    minutos_decimales = (grados_decimales - grados) * 60
    minutos = int(minutos_decimales)
    segundos = (minutos_decimales - minutos) * 60
    return f"{grados}° {minutos}' {segundos:.8f}\" {direccion}"

def gms_a_decimal(grados:int, minutos:int, segundos:float, direccion:str, tipo:str):
    # Validaciones básicas de rango
    if tipo == "lat":
        if not (0 <= abs(grados) <= 90):
            raise ValueError("Grados de latitud fuera de rango (0–90).")
    else:
        if not (0 <= abs(grados) <= 180):
            raise ValueError("Grados de longitud fuera de rango (0–180).")
    if not (0 <= minutos < 60):
        raise ValueError("Minutos fuera de rango (0–59).")
    if not (0 <= segundos < 60):
        raise ValueError("Segundos fuera de rango (0–59.999...).")
    if tipo == "lat" and direccion not in ("N","S"):
        raise ValueError("Dirección de latitud debe ser N o S.")
    if tipo == "lon" and direccion not in ("E","W"):
        raise ValueError("Dirección de longitud debe ser E o W.")

    decimal = abs(grados) + minutos/60 + segundos/3600
    if direccion in ("S","W"):
        decimal = -decimal
    return decimal

# ---------- FUNCIONES DE CÁLCULO ----------
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

# ---------- ENTRADA DE COORDENADAS (MODO SELECCIONABLE) ----------
st.markdown("#### Formato de coordenadas de entrada")
modo_coord = st.radio(
    "Formato de coordenadas de entrada",
    ["Decimal", "Grados, Minutos y Segundos (GMS)"],
    horizontal=True
)

def input_decimal(label_lat="Latitud inicial (decimal)", label_lon="Longitud inicial (decimal)"):
    c1, c2 = st.columns(2)
    with c1:
        lat_txt = st.text_input(label_lat, value="8.8066", key=f"{categoria}_lat_dec")
    with c2:
        lon_txt = st.text_input(label_lon, value="-82.5403", key=f"{categoria}_lon_dec")
    try:
        lat_val = float(lat_txt)
        lon_val = float(lon_txt)
    except ValueError:
        st.error("Por favor ingresa números válidos para latitud y longitud en decimal.")
        st.stop()
    # Mostrar conversión a GMS
    st.caption(f"**GMS:** Lat {decimal_a_gms(lat_val,'lat')}  |  Lon {decimal_a_gms(lon_val,'lon')}")
    return lat_val, lon_val

def input_gms():
    st.write("**Latitud (GMS)**")
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        lat_g = st.number_input("Grados (lat)", value=8, step=1, format="%d")
    with c2:
        lat_m = st.number_input("Minutos (lat)", value=48, min_value=0, max_value=59, step=1, format="%d")
    with c3:
        lat_s = st.number_input("Segundos (lat)", value=23.76, min_value=0.0, max_value=59.999999, step=0.01, format="%.6f")
    with c4:
        lat_dir = st.selectbox("Dir (lat)", options=["N","S"], index=0)

    st.write("**Longitud (GMS)**")
    c5, c6, c7, c8 = st.columns([1,1,1,1])
    with c5:
        lon_g = st.number_input("Grados (lon)", value=82, step=1, format="%d")
    with c6:
        lon_m = st.number_input("Minutos (lon)", value=32, min_value=0, max_value=59, step=1, format="%d")
    with c7:
        lon_s = st.number_input("Segundos (lon)", value=25.08, min_value=0.0, max_value=59.999999, step=0.01, format="%.6f")
    with c8:
        lon_dir = st.selectbox("Dir (lon)", options=["E","W"], index=1)

    try:
        lat_val = gms_a_decimal(lat_g, lat_m, lat_s, lat_dir, "lat")
        lon_val = gms_a_decimal(lon_g, lon_m, lon_s, lon_dir, "lon")
    except Exception as e:
        st.error(f"Error en GMS: {e}")
        st.stop()

    # Mostrar conversión a decimal
    st.caption(f"**Decimal:** Lat {lat_val:.10f}  |  Lon {lon_val:.10f}")
    return lat_val, lon_val

# Obtener coordenadas según el modo elegido (ambas devuelven en decimal para cálculos)
if modo_coord == "Decimal":
    lat, lon = input_decimal()
else:
    lat, lon = input_gms()

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
        except Exception as e:
            st.error(f"Error en los acimuts ingresados: {e}")

elif categoria == "Calculo de distancia":
    col1, col2 = st.columns(2)
    with col1:
        lat2_mode = st.radio("Formato coord. punto 2", ["Decimal","GMS"], horizontal=True, key="mode_p2")
    with col2:
        st.write(" ")

    if lat2_mode == "Decimal":
        c1, c2 = st.columns(2)
        with c1:
            lat2_txt = st.text_input("Latitud 2 (decimal)", value="8.8066", key="lat2")
        with c2:
            lon2_txt = st.text_input("Longitud 2 (decimal)", value="-82.5403", key="lon2")
        try:
            lat2 = float(lat2_txt)
            lon2 = float(lon2_txt)
        except ValueError:
            st.error("Ingresa lat/lon válidos para el punto 2 (decimal).")
            st.stop()
        st.caption(f"GMS pto2: Lat {decimal_a_gms(lat2,'lat')} | Lon {decimal_a_gms(lon2,'lon')}")
    else:
        st.write("**Punto 2 (GMS)**")
        c1, c2, c3, c4 = st.columns([1,1,1,1])
        with c1:
            lat2_g = st.number_input("Grados (lat2)", value=8, step=1, format="%d")
        with c2:
            lat2_m = st.number_input("Min (lat2)", value=48, min_value=0, max_value=59, step=1, format="%d")
        with c3:
            lat2_s = st.number_input("Seg (lat2)", value=23.76, min_value=0.0, max_value=59.999999, step=0.01, format="%.6f")
        with c4:
            lat2_dir = st.selectbox("Dir (lat2)", options=["N","S"], index=0)
        d1, d2, d3, d4 = st.columns([1,1,1,1])
        with d1:
            lon2_g = st.number_input("Grados (lon2)", value=82, step=1, format="%d")
        with d2:
            lon2_m = st.number_input("Min (lon2)", value=32, min_value=0, max_value=59, step=1, format="%d")
        with d3:
            lon2_s = st.number_input("Seg (lon2)", value=25.08, min_value=0.0, max_value=59.999999, step=0.01, format="%.6f")
        with d4:
            lon2_dir = st.selectbox("Dir (lon2)", options=["E","W"], index=1)
        try:
            lat2 = gms_a_decimal(lat2_g, lat2_m, lat2_s, lat2_dir, "lat")
            lon2 = gms_a_decimal(lon2_g, lon2_m, lon2_s, lon2_dir, "lon")
        except Exception as e:
            st.error(f"Error en GMS del punto 2: {e}")
            st.stop()
        st.caption(f"Decimal pto2: Lat {lat2:.10f} | Lon {lon2:.10f}")

    if st.button("Calcular distancia y acimut", key="calc_dist"):
        try:
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
        except Exception as e:
            st.error(f"Error en el cálculo: {e}")

elif categoria == "Calculo de distancia central":
    num_puntos = st.number_input("Número de coordenadas desde el punto central", value=2, min_value=1)
    puntos = []
    for i in range(num_puntos):
        modo_p = st.radio(f"Formato coord. punto {i+1}", ["Decimal","GMS"], horizontal=True, key=f"mode_central_{i}")
        if modo_p == "Decimal":
            c1, c2 = st.columns(2)
            with c1:
                lat_p = st.text_input(f"Latitud punto {i+1} (decimal)", value="8.8066", key=f"lat_central_{i}")
            with c2:
                lon_p = st.text_input(f"Longitud punto {i+1} (decimal)", value="-82.5403", key=f"lon_central_{i}")
            try:
                lat_f = float(lat_p); lon_f = float(lon_p)
            except ValueError:
                st.error(f"Ingreso inválido para punto {i+1} (decimal).")
                st.stop()
            st.caption(f"GMS pto{i+1}: Lat {decimal_a_gms(lat_f,'lat')} | Lon {decimal_a_gms(lon_f,'lon')}")
        else:
            st.write(f"**Punto {i+1} (GMS)**")
            c1, c2, c3, c4 = st.columns([1,1,1,1])
            with c1:
                g = st.number_input(f"Grados (lat {i+1})", value=8, step=1, format="%d", key=f"latg_{i}")
            with c2:
                m = st.number_input(f"Min (lat {i+1})", value=48, min_value=0, max_value=59, step=1, format="%d", key=f"latm_{i}")
            with c3:
                s = st.number_input(f"Seg (lat {i+1})", value=23.76, min_value=0.0, max_value=59.999999, step=0.01, format="%.6f", key=f"lats_{i}")
            with c4:
                d = st.selectbox(f"Dir (lat {i+1})", options=["N","S"], index=0, key=f"latd_{i}")
            d1, d2, d3, d4 = st.columns([1,1,1,1])
            with d1:
                g2 = st.number_input(f"Grados (lon {i+1})", value=82, step=1, format="%d", key=f"long_{i}")
            with d2:
                m2 = st.number_input(f"Min (lon {i+1})", value=32, min_value=0, max_value=59, step=1, format="%d", key=f"lonm_{i}")
            with d3:
                s2 = st.number_input(f"Seg (lon {i+1})", value=25.08, min_value=0.0, max_value=59.999999, step=0.01, format="%.6f", key=f"lons_{i}")
            with d4:
                d2dir = st.selectbox(f"Dir (lon {i+1})", options=["E","W"], index=1, key=f"lond_{i}")
            try:
                lat_f = gms_a_decimal(g, m, s, d, "lat")
                lon_f = gms_a_decimal(g2, m2, s2, d2dir, "lon")
            except Exception as e:
                st.error(f"Error en GMS del punto {i+1}: {e}")
                st.stop()
            st.caption(f"Decimal pto{i+1}: Lat {lat_f:.10f} | Lon {lon_f:.10f}")
        puntos.append((lat_f, lon_f))

    if st.button("Calcular distancias centrales", key="calc_central"):
        resultados = []
        try:
            for i, (lat_f, lon_f) in enumerate(puntos, start=1):
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
        except Exception as e:
            st.error(f"Error en el cálculo: {e}")

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
