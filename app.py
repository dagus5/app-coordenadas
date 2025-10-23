def mostrar_mapa(df, lat=None, lon=None, categoria=None):
    # Verificación de coordenadas iniciales
    if lat is None or lon is None:
        if "Latitud central" in df.columns and "Longitud central" in df.columns:
            lat = float(df["Latitud central"].iloc[0])
            lon = float(df["Longitud central"].iloc[0])
        elif "Latitud 1" in df.columns and "Longitud 1" in df.columns:
            lat = float(df["Latitud 1"].iloc[0])
            lon = float(df["Longitud 1"].iloc[0])
        elif "Latitud punto" in df.columns and "Longitud punto" in df.columns:
            lat = float(df["Latitud punto"].iloc[0])
            lon = float(df["Longitud punto"].iloc[0])
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

    st_folium(mapa, width=700, height=500)

