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
    df_total_anomalous_baye = pd.read_csv(os.path.join(modelo_dir, "df_total_anomalous_baye.csv"))


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

    numero_de_datos = 4100


    if st.session_state.get("modelo_1") is not None:
        try:
            st.session_state.datos_modelo_1 = st.session_state.modelo_1.sample(num_rows=numero_de_datos)
        except Exception as e:
            st.sidebar.error(f"❌ Error al generar datos para Modelo 1: {e}")

    if st.session_state.get("modelo_2") is not None:
        try:
            st.session_state.datos_modelo_2 = st.session_state.modelo_2.sample(num_rows=numero_de_datos)
            if "XX012" in st.session_state.datos_modelo_2.columns:
                st.session_state.datos_modelo_2["XX012"] *= -1
        except Exception as e:
            st.sidebar.error(f"❌ Error al generar datos para Modelo 2: {e}")

    if st.session_state.get("modelo_3") is not None:
        try:
            n1 = int(numero_de_datos * 0.8)
            n2 = numero_de_datos - n1  # asegurar total exacto

            # ✅ Tomar los datos ya generados previamente
            datos_1 = st.session_state.datos_modelo_1.sample(n=n1)
            datos_2 = st.session_state.datos_modelo_2.sample(n=n2)

            datos_combinados = pd.concat([datos_1, datos_2]).sample(frac=1).reset_index(drop=True)

            st.session_state.datos_modelo_3 = datos_combinados

        except Exception as e:
            st.sidebar.error(f"❌ Error al generar datos para Modelo 3: {e}")
        # Botón para forzar recarga
        # Reemplazar columnas específicas en datos_modelo_2 por valores del dataframe externo
    columnas_reemplazo = [
        "TE101", "TE201", "TE202", "TE272", "RPM", "TE511", "TE517", "TE5011A","TE600 - Carga"
    ]

    try:
        # Asegurar que df_total_anomalous_baye tiene suficientes filas
        if df_total_anomalous_baye.shape[0] >= st.session_state.datos_modelo_2.shape[0]:
            reemplazo = df_total_anomalous_baye[columnas_reemplazo].sample(
                n=st.session_state.datos_modelo_2.shape[0]
            ).reset_index(drop=True)

            st.session_state.datos_modelo_2.loc[:, columnas_reemplazo] = reemplazo.values
        else:
            st.sidebar.warning("⚠️ No hay suficientes datos en df_total_anomalous_baye para reemplazo.")
    except Exception as e:
        st.sidebar.error(f"❌ Error al reemplazar columnas en Modelo 2: {e}")

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
