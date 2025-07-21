import math
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
from Secciones.visualización import *

def cargar_modelos():
    import os
    import joblib
    import streamlit as st

    # st.sidebar.subheader("📂 Carga automática de modelos desde la carpeta /modelos")

    modelo_dir = os.path.join(os.getcwd(), "modelos")

    # Carga de modelos
    modelo_1_path = os.path.join(modelo_dir, "VAE_Normal.pkl")
    modelo_2_path = os.path.join(modelo_dir, "VAE_Anomalías.pkl")
    modelo_3_path = os.path.join(modelo_dir, "VAE_Operación.pkl")

    # Modelo 1: VAE_Normal
    if os.path.exists(modelo_1_path):
        try:
            st.session_state.modelo_1 = joblib.load(modelo_1_path)
        except Exception as e:
            st.sidebar.error(f"❌ Error al cargar VAE_Normal.pkl: {e}")
    else:
        st.sidebar.warning("⚠️ No se encontró VAE_Normal.pkl en /modelos")

    # Modelo 2: VAE_Anomalías
    if os.path.exists(modelo_2_path):
        try:
            st.session_state.modelo_2 = joblib.load(modelo_2_path)
        except Exception as e:
            st.sidebar.error(f"❌ Error al cargar VAE_Anomalías.pkl: {e}")
    else:
        st.sidebar.warning("⚠️ No se encontró VAE_Anomalías.pkl en /modelos")

    # Modelo 3: VAE_Operación
    if os.path.exists(modelo_3_path):
        try:
            st.session_state.modelo_3 = joblib.load(modelo_3_path)
        except Exception as e:
            st.sidebar.error(f"❌ Error al cargar VAE_Operación.pkl: {e}")
    else:
        st.sidebar.warning("⚠️ No se encontró VAE_Operación.pkl en /modelos")

    # Parámetros de generación
    MIN_FILAS = 7 * 24 * 4 * 4  # 2688

    numero_de_datos = st.sidebar.number_input(
        "Mediciones a generar",
        min_value=MIN_FILAS,
        max_value=10000,
        value=MIN_FILAS,
        step=24
    )

    if st.sidebar.button("🔄 Generar datos"):
        if st.session_state.get("modelo_1") is not None:
            try:
                st.session_state.datos_modelo_1 = st.session_state.modelo_1.sample(num_rows=numero_de_datos)
            except Exception as e:
                st.sidebar.error(f"❌ Error al generar datos para Modelo 1: {e}")

        if st.session_state.get("modelo_2") is not None:
            try:
                st.session_state.datos_modelo_2 = st.session_state.modelo_2.sample(num_rows=numero_de_datos)
            except Exception as e:
                st.sidebar.error(f"❌ Error al generar datos para Modelo 2: {e}")

        if st.session_state.get("modelo_3") is not None:
            try:
                st.session_state.datos_modelo_3 = st.session_state.modelo_3.sample(num_rows=numero_de_datos)
            except Exception as e:
                st.sidebar.error(f"❌ Error al generar datos para Modelo 3: {e}")


    # Vista previa y descarga
    visualizar_subsistemas()

    # if st.session_state.get("datos_modelo_1") is not None:
    #     st.subheader("🟢 Datos sin anomalías")
    #     st.dataframe(st.session_state.datos_modelo_1)
    #     csv_1 = st.session_state.datos_modelo_1.to_csv(index=False).encode("utf-8")
    #     st.download_button(
    #         label="⬇️ Descargar CSV Modelo 1",
    #         data=csv_1,
    #         file_name="datos_sinteticos_modelo1.csv",
    #         mime="text/csv"
    #     )

    # if st.session_state.get("datos_modelo_2") is not None:
    #     st.subheader("🔴 Datos con anomalías")
    #     st.dataframe(st.session_state.datos_modelo_2)
    #     csv_2 = st.session_state.datos_modelo_2.to_csv(index=False).encode("utf-8")
    #     st.download_button(
    #         label="⬇️ Descargar CSV Modelo 2",
    #         data=csv_2,
    #         file_name="datos_sinteticos_modelo2.csv",
    #         mime="text/csv"
    #     )
