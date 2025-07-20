import streamlit as st
import json
import math
from utils.graficas import graficar_modelos_comparados

# Cargar subsistemas
with open("subsistemas.json", "r", encoding="utf-8") as f:
    subsistemas = json.load(f)

# Cargar resultado.json (con límites y mediciones)
with open("resultado.json", "r", encoding="utf-8") as f:
    resultado = json.load(f)

def visualizar_subsistemas():
    st.header("📈 Visualización por subsistema")

    # Verificamos que haya datos generados
    df_1 = st.session_state.get("datos_modelo_1")
    df_2 = st.session_state.get("datos_modelo_2")

    if df_1 is None and df_2 is None:
        st.info("Primero debes generar datos en la sección de generación.")
        return
    
    # Determinar columnas disponibles en ambos modelos
    columnas_disponibles = []
    if df_1 is not None:
        columnas_disponibles.extend(df_1.columns)
    if df_2 is not None:
        columnas_disponibles.extend(df_2.columns)
    columnas_disponibles = list(set(columnas_disponibles))

    # --- CONTROLES EN EL SIDEBAR ---
    with st.sidebar:
        st.subheader("🧭 Visualización de subsistemas")

        # Selección de subsistema
        subsistema_sel = st.selectbox("🔧 Subsistema", list(subsistemas.keys()))
        variables_disponibles = [v for v in subsistemas[subsistema_sel] if v in columnas_disponibles]

        # Selección de variables
        seleccionadas = st.multiselect("📌 Variables a graficar", variables_disponibles, default=variables_disponibles)

        # Número de muestras aleatorias a visualizar
        n_muestras=76
        # n_muestras = st.slider("🎯 Número de datos a graficar (aleatorios)", min_value=12, max_value=48, value=24, step=12)

        # Mostrar u ocultar cada modelo
        mostrar_datos_reales = st.checkbox("🟣 Mostrar datos reales", value=True)
        mostrar_modelo_1 = st.checkbox("🟢 Mostrar datos normales", value=False)
        mostrar_modelo_2 = st.checkbox("🔴 Mostrar datos con anomalías", value=False)
        


        # Botón para forzar recarga
        if st.button("🔄 Recargar muestra"):
            st.session_state["recargar_visualizacion"] = not st.session_state.get("recargar_visualizacion", False)

    # --- VISUALIZACIÓN ---
    if not seleccionadas:
        st.warning("Selecciona al menos una variable para graficar.")
        return

    # Subset aleatorio de datos
    df_1_muestra = df_1[seleccionadas].sample(n=n_muestras) if mostrar_modelo_1 and df_1 is not None else None
    df_2_muestra = df_2[seleccionadas].sample(n=n_muestras) if mostrar_modelo_2 and df_2 is not None else None

    st.caption(f"🎲 Mostrando {n_muestras} muestras aleatorias por variable.")
    limites = {
        "PT101": (0, 10),
        "PT401": (0, 7),
        "RPM": (600, 1100),
        "TE201": (20, 90),
        "TE600 - Aire entrada al turbo": (30, 60),
        "PT201": (0, 8),
        "TE202": (40, 100),
        "PDT243": (0, 2),
        "PT271": (0, 8),
    }


    graficar_modelos_comparados(
        resultado,
        df_1=df_1_muestra,
        df_2=df_2_muestra,
        seleccionadas=seleccionadas,
        titulo=f"Visualización - {subsistema_sel}",
        mostrar_reales=mostrar_datos_reales,
        limites=limites
    )