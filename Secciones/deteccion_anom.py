import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import joblib
from matplotlib.ticker import MaxNLocator
from sklearn.linear_model import LinearRegression
import math
import random
from utils.anom import anomalias

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
def detec_A():
    st.header("📈 Detección de anomalías")
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("colorblind")
     
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("colorblind")

    df_1 = st.session_state.get("datos_modelo_1")
    df_2 = st.session_state.get("datos_modelo_2")
    df_3 = st.session_state.get("datos_modelo_3")  # ✅ Nuevo modelo
    resultado = st.session_state.get("resultado")
    subsistemas = st.session_state.get("subsistemas")

    if not resultado or all(df is None for df in [df_1, df_2, df_3]):
        st.warning("⚠️ Asegúrate de haber generado datos y cargado el diccionario `resultado`.")
        st.stop()

    st.sidebar.subheader("⚙️ Simulación")
    modelo = st.sidebar.selectbox(
        "🟦 Tipo de datos a visualizar",
        ["🟢 Datos sin anomalías", "🔴 Datos con anomalías"]
    )

    tipo_datos=modelo
    if "🟢 Datos sin anomalías" in modelo:
        df = df_1
    elif "🔴 Datos con anomalías" in modelo:
        df = df_2
    else:
        df = df_3

    if df is None or not isinstance(df, pd.DataFrame):
        st.warning(f"No se encontraron datos para {modelo}.")
        st.stop()


    subsistema_sel = st.sidebar.selectbox("Subsistema", list(subsistemas.keys()))
    variables_disponibles = [v for v in subsistemas[subsistema_sel] if v in df.columns and v in resultado]



    # === 2. Umbrales específicos por subsistema ===
    umbrales = {
        "Sistema de Refrigeración": 9,
        "Sistema de Combustible": 0.36,
        "Sistema de Lubricación": 1,
        "Temperatura de Gases de Escape": 3.5
    }

    umbral = float(umbrales[subsistema_sel])
    if not variables_disponibles:
        st.warning("No hay variables válidas para este subsistema.")
        st.stop()


    st.sidebar.markdown("📌 Variables del subsistema:")
    var_sel= []

    for var in variables_disponibles:
        nombre_largo = resultado[var].get("Nombre", var)

        # Crear el texto con tooltip
        label_html = f"""
            <label title="{nombre_largo}" style="cursor: pointer;">
                {var}
            </label>
        """

        # Usamos una columna para alinear checkbox sin texto, y el HTML al lado
        col1, col2 = st.sidebar.columns([1, 4])
        
        with col1:
            checked = st.checkbox("", value=True, key=var)
        
        with col2:
            st.markdown(label_html, unsafe_allow_html=True)

        if checked:
            var_sel.append(var)
    iteraciones = 1
    velocidad = 0.01



    # Lista de subsistemas con modelo asociado
    subsistemas_modelados = {
        "Sistema de Combustible": "combustible",
        "Temperatura de Gases de Escape": "gases",
        "Sistema de Lubricación": "lubricante",
        "Sistema de Refrigeración": "refrigeracion"
        
    }

        # === 2. Umbrales específicos por subsistema ===
    umbrales = {
        "Sistema de Refrigeración": 9,
        "Sistema de Combustible": 18,
        "Sistema de Lubricación": 1,
        "Temperatura de Gases de Escape": 3.5
    }
    umbral = float(umbrales[subsistema_sel])

    print("Subsistema seleccionado por el usuario:", subsistema_sel)
    modelo_dir = os.path.join(os.getcwd(), "modelos")
    # Diccionario para guardar los modelos y escaladores por subsistema
    modelos = {}
    scalers = {}
    # Cargar modelos y scalers
    for nombre_sis, sufijo in subsistemas_modelados.items():
        try:
            modelo_path = os.path.join(modelo_dir, f"modelo_lstm_vae_{sufijo}.h5")
            scaler_path = os.path.join(modelo_dir, f"scaler_lstm_vae_{sufijo}.pkl")

            print(f"🔄 Cargando modelo para {nombre_sis} desde {modelo_path}")
            print(f"🔄 Cargando scaler para {nombre_sis} desde {scaler_path}")

            modelos[nombre_sis] = load_model(modelo_path, compile=False)
            scalers[nombre_sis] = joblib.load(scaler_path)

            print(f"✅ Modelo y scaler cargados para: {nombre_sis}")


        except Exception as e:
            print(f"❌ Error al cargar modelo o scaler de {nombre_sis}: {e}")

            st.stop()

    print("📁 Modelos disponibles:", list(modelos.keys()))


    # Seleccionar el modelo según lo que escogió el usuario
    if subsistema_sel in modelos:
        print(f"✅ Se encontró el modelo para {subsistema_sel}")

        modelo = modelos[subsistema_sel]
        scaler = scalers[subsistema_sel]
    else:
        print(f"⚠️ Subsistema no tiene modelo: {subsistema_sel}")
        st.warning("🔍 Este subsistema aún no tiene un modelo asociado.")
        # Límites de cada variable
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
    # Inicializa la muestra si no existe
    n_datos=30
    if "df_seleccion" not in st.session_state:
        st.session_state.df_seleccion = df[variables_disponibles].sample(n=n_datos)

    # Botón para recargar muestra
    if st.sidebar.button("🔄 Recargar muestra"):
        st.session_state.df_seleccion = df[variables_disponibles].sample(n=n_datos)

    # Llamada a la función anomalias con la muestra ya tomada
    anomalias(
        df=st.session_state.df_seleccion,
        modelo=modelo,
        scaler=scaler,
        variables_disponibles=var_sel,
        resultado=resultado,
        subsistema_sel=subsistema_sel,
        limites=limites
    )