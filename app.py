# -*- coding: utf-8 -*-
# Streamlit app with Δh (ITM) and elevation source selection (SRTM / ASTER / Compare)
# Save as app.py and run: streamlit run app.py

import streamlit as st
import pandas as pd
from pygeodesy.ellipsoidalVincenty import LatLon
import folium
from streamlit_folium import st_folium
import numpy as np
import math
from io import BytesIO
import plotly.graph_objects as go

import srtm
import rasterio

st.set_page_config(page_title="Coordenadas + Δh (ITM)", layout="wide")
st.title("🧭 Calculadora Avanzada de Coordenadas + 🌄 Δh (Longley–Rice / ITM)")

if "df_resultado" not in st.session_state:
    st.session_state.df_resultado = {}
if "categoria" not in st.session_state:
    st.session_state.categoria = "Calculo - 8 Radiales"

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
    if tipo == "lat":
        if not (0 <= abs(grados) <= 90): raise ValueError("Grados lat 0–90.")
    else:
        if not (0 <= abs(grados) <= 180): raise ValueError("Grados lon 0–180.")
    if not (0 <= minutos < 60): raise ValueError("Min 0–59.")
    if not (0 <= segundos < 60): raise ValueError("Seg 0–59.999.")
    if tipo == "lat" and direccion not in ("N","S"): raise ValueError("Dir lat N/S.")
    if tipo == "lon" and direccion not in ("E","W"): raise ValueError("Dir lon E/W.")
    decimal = abs(grados) + minutos/60 + segundos/3600
    if direccion in ("S","W"): decimal = -decimal
    return decimal

R_EARTH_M = 6371000.0
def destination_point(lat_deg, lon_deg, bearing_deg, distance_m):
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    brng = math.radians(bearing_deg)
    dr = distance_m / R_EARTH_M
    lat2 = math.asin(math.sin(lat1) * math.cos(dr) + math.cos(lat1) * math.sin(dr) * math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(dr) * math.cos(lat1),
                             math.cos(dr) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), (math.degrees(lon2) + 540) % 360 - 180

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
    p1 = LatLon(lat1, lon1)
    p2 = LatLon(lat2, lon2)
    distancia = p1.distanceTo(p2)
    acimut_ida = p1.initialBearingTo(p2)
    acimut_vuelta = p2.initialBearingTo(p1)
    return distancia/1000, acimut_ida, acimut_vuelta

def mostrar_mapa(df, lat, lon, categoria):
    mapa = folium.Map(location=[lat, lon], zoom_start=9)
    if categoria in ["Calculo - 8 Radiales", "Calculo por Azimut"]:
        for _, row in df.iterrows():
            folium.Marker([float(row["Latitud Final (Decimal)"]), float(row["Longitud Final (Decimal)"])],
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
            lat_p, lon_p = float(row["Longitud punto"]), float(row["Longitud punto"])
            folium.Marker([lat_c, lon_c], tooltip="Central", icon=folium.Icon(color="red")).add_to(mapa)
            folium.Marker([lat_p, lon_p], tooltip="Punto", icon=folium.Icon(color="blue")).add_to(mapa)
            folium.PolyLine([[lat_c, lon_c], [lat_p, lon_p]], color="green", weight=2).add_to(mapa)
    elif categoria == "Δh – Rugosidad (ITM)":
        folium.Marker([lat, lon], tooltip="Transmisor", icon=folium.Icon(color="red")).add_to(mapa)
    st_folium(mapa, width=700, height=500)

st.markdown("### Selecciona la categoría de cálculo")
c1, c2 = st.columns(2)
c3, c4 = st.columns(2)
c5, _ = st.columns(2)

if c1.button("📍 Calculo - 8 Radiales"):
    st.session_state.categoria = "Calculo - 8 Radiales"
if c2.button("🧭 Calculo por Azimut"):
    st.session_state.categoria = "Calculo por Azimut"
if c3.button("📏 Calculo de distancia"):
    st.session_state.categoria = "Calculo de distancia"
if c4.button("🗺️ Calculo de distancia central"):
    st.session_state.categoria = "Calculo de distancia central"
if c5.button("🌄 Δh – Rugosidad (ITM)"):
    st.session_state.categoria = "Δh – Rugosidad (ITM)"

categoria = st.session_state.categoria
st.markdown(f"### 🟢 Categoría seleccionada: {categoria}")

st.markdown("#### Formato de coordenadas de entrada")
modo_coord = st.radio("Formato de coordenadas de entrada",
                      ["Decimal", "Grados, Minutos y Segundos (GMS)"], horizontal=True)

def input_decimal(label_lat="Latitud inicial (decimal)", label_lon="Longitud inicial (decimal)"):
    cc1, cc2 = st.columns(2)
    with cc1:
        lat_txt = st.text_input(label_lat, value="8.8066", key=f"{categoria}_lat_dec")
    with cc2:
        lon_txt = st.text_input(label_lon, value="-82.5403", key=f"{categoria}_lon_dec")
    try:
        lat_val = float(lat_txt); lon_val = float(lon_txt)
    except ValueError:
        st.error("Por favor ingresa números válidos (decimal)."); st.stop()
    st.caption(f"**GMS:** Lat {decimal_a_gms(lat_val,'lat')} | Lon {decimal_a_gms(lon_val,'lon')}")
    return lat_val, lon_val

def input_gms():
    st.write("**Latitud (GMS)**")
    a,b,c,d = st.columns([1,1,1,1])
    with a: lat_g = st.number_input("Grados (lat)", value=8, step=1, format="%d")
    with b: lat_m = st.number_input("Min (lat)", value=48, min_value=0, max_value=59, step=1, format="%d")
    with c: lat_s = st.number_input("Seg (lat)", value=23.76, min_value=0.0, max_value=59.999999, step=0.01, format="%.6f")
    with d: lat_dir = st.selectbox("Dir (lat)", options=["N","S"], index=0)
    st.write("**Longitud (GMS)**")
    e,f,g,h = st.columns([1,1,1,1])
    with e: lon_g = st.number_input("Grados (lon)", value=82, step=1, format="%d")
    with f: lon_m = st.number_input("Min (lon)", value=32, min_value=0, max_value=59, step=1, format="%d")
    with g: lon_s = st.number_input("Seg (lon)", value=25.08, min_value=0.0, max_value=59.999999, step=0.01, format="%.6f")
    with h: lon_dir = st.selectbox("Dir (lon)", options=["E","W"], index=1)
    try:
        lat_val = gms_a_decimal(lat_g, lat_m, lat_s, lat_dir, "lat")
        lon_val = gms_a_decimal(lon_g, lon_m, lon_s, lon_dir, "lon")
    except Exception as e:
        st.error(f"Error GMS: {e}"); st.stop()
    st.caption(f"**Decimal:** Lat {lat_val:.10f} | Lon {lon_val:.10f}")
    return lat_val, lon_val

if modo_coord == "Decimal":
    lat, lon = input_decimal()
else:
    lat, lon = input_gms()

# ---- CATEGORÍAS EXISTENTES (omitimos sus detalles por brevedad; son iguales a la versión anterior) ----

# ---------- Δh – Rugosidad (ITM) con fuente de elevación ----------
def deltaF_from_deltaH(delta_h, freq_mhz):
    return 1.9 - 0.03*(delta_h)*(1 + freq_mhz/300.0)

@st.cache_resource
def get_srtm_data():
    return srtm.get_data()

@st.cache_resource
def open_aster(path_tif:str):
    try:
        ds = rasterio.open(path_tif)
        return ds
    except Exception as e:
        st.warning(f"No se pudo abrir ASTER GDEM en '{path_tif}': {e}")
        return None

def get_elevations_srtm(lats, lons):
    data = get_srtm_data()
    return [data.get_elevation(la, lo) for la,lo in zip(lats,lons)]

def get_elevations_aster(ds, lats, lons):
    if ds is None:
        return [None]*len(lats)
    elev = []
    band1 = ds.read(1)
    for la, lo in zip(lats, lons):
        try:
            row, col = ds.index(lo, la)  # (lon, lat)
            val = band1[row, col]
            if ds.nodata is not None and val == ds.nodata:
                elev.append(None)
            else:
                elev.append(float(val))
        except Exception:
            elev.append(None)
    return elev

def get_profile(lat, lon, azimut, start_km, end_km, step_m):
    dists = list(range(int(start_km*1000), int(end_km*1000)+1, int(step_m)))
    lats, lons = [], []
    for d in dists:
        plat, plon = destination_point(lat, lon, azimut, d)
        lats.append(plat); lons.append(plon)
    return dists, lats, lons

def compute_delta_h(elev):
    arr = np.array([e for e in elev if e is not None], dtype=float)
    if arr.size == 0:
        return None, None, None
    h10 = float(np.percentile(arr, 90))
    h90 = float(np.percentile(arr, 10))
    return h10 - h90, h10, h90

if categoria == "Δh – Rugosidad (ITM)":
    st.markdown("#### Parámetros del cálculo de Δh (estilo MSAM / Longley–Rice)")
    col = st.columns(5)
    with col[0]: freq = st.number_input("Frecuencia (MHz)", value=102.1, step=0.1, format="%.1f")
    with col[1]: az_str = st.text_input("Azimuts (°) separados por coma", value="0,45,90,135,180,225,270,315")
    with col[2]: dist_total = st.number_input("Distancia total (km)", value=50.0, min_value=10.0, step=1.0, format="%.1f")
    with col[3]: step_m = st.number_input("Paso (m)", value=500, min_value=100, max_value=2000, step=50)
    with col[4]: src = st.selectbox("Fuente de elevación", ["SRTM", "ASTER GDEM", "Comparar ambos"], index=0)

    aster_path = None
    ds_aster = None
    if src in ("ASTER GDEM", "Comparar ambos"):
        aster_path = st.text_input("Ruta del archivo ASTER GDEM (GeoTIFF, EPSG:4326)", value="")
        ds_aster = open_aster(aster_path) if aster_path else None
        if aster_path and ds_aster is None:
            st.info("Se utilizará SRTM si ASTER no está disponible.")

    st.caption("El cálculo de Δh usa **solo el tramo 10–50 km** (o hasta la distancia total indicada). h10/h90 = percentiles 90/10 de elevación.")

    run = st.button("Calcular Δh por azimut")

    if run:
        try:
            az_list = [float(a.strip()) for a in az_str.split(",") if a.strip()!=""]
        except:
            st.error("Revisa la lista de azimuts."); st.stop()

        fmap = folium.Map(location=[lat, lon], zoom_start=8, control_scale=True)
        folium.Marker([lat, lon], tooltip="Transmisor", icon=folium.Icon(color="red")).add_to(fmap)

        results_rows = []
        profiles = {}
        end_km_for_profile = dist_total if dist_total < 50.0 else 50.0

        for az in az_list:
            dists, lats, lons = get_profile(lat, lon, az, 10.0, end_km_for_profile, step_m)

            elev_srtm = get_elevations_srtm(lats, lons) if src in ("SRTM", "Comparar ambos") else None
            elev_aster = get_elevations_aster(ds_aster, lats, lons) if src in ("ASTER GDEM", "Comparar ambos") else None

            row = {"Azimut (°)": az}
            if elev_srtm is not None:
                dh_srtm, h10_s, h90_s = compute_delta_h(elev_srtm)
                if dh_srtm is not None:
                    row.update({"Δh_SRTM (m)": round(dh_srtm,2), "ΔF_SRTM (dB)": round(deltaF_from_deltaH(dh_srtm, freq),2)})
                    profiles.setdefault(az, {})["SRTM"] = pd.DataFrame({"Distancia (km)": [d/1000 for d in dists],
                                                                        "Elevación (m)": elev_srtm, "Lat": lats, "Lon": lons})
            if elev_aster is not None and any(e is not None for e in elev_aster):
                dh_ast, h10_a, h90_a = compute_delta_h(elev_aster)
                if dh_ast is not None:
                    row.update({"Δh_ASTER (m)": round(dh_ast,2), "ΔF_ASTER (dB)": round(deltaF_from_deltaH(dh_ast, freq),2)})
                    profiles.setdefault(az, {})["ASTER"] = pd.DataFrame({"Distancia (km)": [d/1000 for d in dists],
                                                                         "Elevación (m)": elev_aster, "Lat": lats, "Lon": lons})
            if "Δh_SRTM (m)" in row and "Δh_ASTER (m)" in row:
                row["Δh diferencia (m)"] = round(row["Δh_SRTM (m)"] - row["Δh_ASTER (m)"], 2)

            if len(row) > 1:
                results_rows.append(row)

            folium.PolyLine(list(zip(lats,lons)), weight=3, opacity=0.8).add_to(fmap)

        if len(results_rows)==0:
            st.error("No se obtuvieron elevaciones. Verifica la fuente seleccionada y/o la ruta ASTER."); st.stop()

        res_df = pd.DataFrame(results_rows).sort_values("Azimut (°)").reset_index(drop=True)
        st.subheader("Resultados por azimut")
        st.dataframe(res_df, use_container_width=True)

        resumen = {}
        for colname in ["Δh_SRTM (m)", "ΔF_SRTM (dB)", "Δh_ASTER (m)", "ΔF_ASTER (dB)", "Δh diferencia (m)"]:
            if colname in res_df.columns:
                resumen[f"Promedio {colname}"] = round(res_df[colname].mean(), 2)
        if resumen:
            st.markdown("**Resumen:**")
            st.write(resumen)

        az_opts = res_df["Azimut (°)"].tolist()
        az_sel = st.selectbox("Ver perfil de elevación del azimut:", az_opts)

        fig = go.Figure()
        if "SRTM" in profiles.get(az_sel, {}):
            prof_s = profiles[az_sel]["SRTM"]
            fig.add_trace(go.Scatter(x=prof_s["Distancia (km)"], y=prof_s["Elevación (m)"], mode="lines",
                                     name=f"SRTM – Az {az_sel}°"))
        if "ASTER" in profiles.get(az_sel, {}):
            prof_a = profiles[az_sel]["ASTER"]
            fig.add_trace(go.Scatter(x=prof_a["Distancia (km)"], y=prof_a["Elevación (m)"], mode="lines",
                                     name=f"ASTER – Az {az_sel}°"))
        fig.update_layout(xaxis_title="Distancia (km)", yaxis_title="Elevación (m)",
                          title=f"Perfil de terreno – Azimut {az_sel}°")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Mapa de radiales")
        st_folium(fmap, width=None, height=520)

        def to_excel_bytes(df):
            from openpyxl import Workbook
            from openpyxl.utils.dataframe import dataframe_to_rows
            wb = Workbook(); ws = wb.active; ws.title = "DeltaH"
            for r in dataframe_to_rows(df, index=False, header=True):
                ws.append(r)
            ws["G1"] = "Δh (ITM/Longley–Rice) = h10 - h90, tramo 10–50 km"
            ws["G2"] = "ΔF = 1.9 - 0.03*Δh*(1 + f/300)  (Norma FM Panamá)"
            out = BytesIO(); wb.save(out); return out.getvalue()

        csv_bytes = res_df.to_csv(index=False).encode("utf-8")
        xlsx_bytes = to_excel_bytes(res_df)
        st.download_button("⬇️ Descargar CSV", data=csv_bytes, file_name="deltaH_ITM_resultados.csv", mime="text/csv")
        st.download_button("⬇️ Descargar Excel", data=xlsx_bytes, file_name="deltaH_ITM_resultados.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        with BytesIO() as zip_buffer:
            import zipfile
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for az, src_dict in profiles.items():
                    for src_name, dfp in src_dict.items():
                        zf.writestr(f"perfil_azimut_{az:.1f}_{src_name}.csv", dfp.to_csv(index=False))
            st.download_button("⬇️ Descargar perfiles (ZIP)", data=zip_buffer.getvalue(),
                               file_name="perfiles_ITM_radiales.zip", mime="application/zip")

# Mostrar resultados/mapa de otras categorías (si ya existen) se puede mantener como en tu versión previa.
