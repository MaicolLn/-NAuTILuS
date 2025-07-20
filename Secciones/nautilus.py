import streamlit as st
import matplotlib.pyplot as plt
import time
import random
import pandas as pd
import streamlit as st
import pandas as pd
import random
import numpy as np
import streamlit as st
import pandas as pd
import random
import seaborn as sns

# === Cargar modelo entrenado y scaler ===
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam
import joblib

# Asegúrate de que estos archivos estén en el mismo directorio que el script o proporciona la ruta completa
from tensorflow.keras.models import load_model
import joblib
import os



modelo_dir = os.path.join(os.getcwd(), "modelos")

# ✅ Cargar sin compilar
model = load_model(os.path.join(modelo_dir, "modelo_lstm_vae.h5"), compile=False)

# Cargar el scaler
scaler = joblib.load(os.path.join(modelo_dir, "scaler_lstm_vae.pkl"))



def nautilus_en_marcha():
    st.header("🚢 Nautilus en marcha")

     # Estilo visual
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("colorblind")

    # === Obtener datos y subsistemas ===
    df_1 = st.session_state.get("datos_modelo_1")
    df_2 = st.session_state.get("datos_modelo_2")
    resultado = st.session_state.get("resultado")
    subsistemas = st.session_state.get("subsistemas")

    if not resultado or (df_1 is None and df_2 is None):
        st.warning("⚠️ Asegúrate de haber generado datos y cargado el diccionario `resultado`.")
        st.stop()

    # === Sidebar con selección de modelo, subsistema y variable ===
    st.sidebar.subheader("⚙️ Simulación")

    modelo = st.sidebar.selectbox("Generación de datos", ["Datos sin anomalías", "Datos con anomalías"])
    df = df_1 if modelo == "Datos sin anomalías" else df_2

    if df is None or not isinstance(df, pd.DataFrame):
        st.warning(f"No se encontraron datos para {modelo}.")
        st.stop()

    subsistema_sel = st.sidebar.selectbox("Subsistema", list(subsistemas.keys()))
    variables_disponibles = [v for v in subsistemas[subsistema_sel] if v in df.columns and v in resultado]

    if not variables_disponibles:
        st.warning("No hay variables válidas para este subsistema.")
        st.stop()

    var_sel = st.sidebar.multiselect("📌 Variable a simular", variables_disponibles, default=variables_disponibles)

    iteraciones = st.sidebar.slider("Días de operación", 2, 20, 5)
    velocidad = st.sidebar.slider("Velocidad de simulación ", 0.01, 2.0, 0.5, 0.1)
    umbral=3

    # Crear contenedores (siempre, al inicio de la sección)
    # Al inicio del módulo o función
    if "contenedor_sim" not in st.session_state:
        st.session_state["contenedor_sim"] = st.empty()
    if "contenedor_health" not in st.session_state:
        st.session_state["contenedor_health"] = st.empty()

    # umbral = st.sidebar.slider("Umbral de anomalía", 0.001, 0.1, 0.01, 0.001)
    iniciar = st.sidebar.button("▶️ Iniciar simulación")

    # === Botón detener escaneo ===
    detener_placeholder = st.sidebar.empty()
    detener = detener_placeholder.button("⏹️ Detener")

    if detener:
        st.session_state["escaneo_activo"] = False

    if not st.session_state.get("escaneo_activo", True):
        if "ultima_fig_sim" in st.session_state:
            st.session_state["contenedor_sim"].pyplot(st.session_state["ultima_fig_sim"])
        if "ultima_fig_hi" in st.session_state:
            st.session_state["contenedor_health"].pyplot(st.session_state["ultima_fig_hi"])



    # === Cargar modelo y scaler ===
    modelo_dir = os.path.join(os.getcwd(), "modelos")
    try:
        model = load_model(os.path.join(modelo_dir, "modelo_lstm_vae.h5"), compile=False)
        scaler = joblib.load(os.path.join(modelo_dir, "scaler_lstm_vae.pkl"))
    except Exception as e:
        st.error(f"❌ Error al cargar modelo o scaler: {e}")
        st.stop()

    # === Iniciar simulación ===
    if iniciar:
        st.session_state["escaneo_activo"] = True

        # Limpiar contenedores anteriores
        st.session_state["contenedor_sim"].empty()
        st.session_state["contenedor_health"].empty()

        # Limpiar figuras anteriores si las guardabas
        st.session_state.pop("ultima_fig_sim", None)
        st.session_state.pop("ultima_fig_hi", None)

        health_index = []
        time_steps = 30

        # Luego las usas normalmente durante la simulación:
        contenedor_sim = st.session_state["contenedor_sim"]
        contenedor_health = st.session_state["contenedor_health"]


        for n in range(iteraciones):
            if not st.session_state.get("escaneo_activo", True):
                st.info("⏹️ Escaneo detenido por el usuario.")
                break

            if len(df) < time_steps:
                st.warning("No hay suficientes datos para la simulación.")
                break

            muestra = df[var_sel].sample(n=time_steps).values * 100
            muestra_2d = muestra.reshape(-1, 1)
            muestra_scaled = scaler.transform(muestra_2d)

            fig_sim, ax_sim = plt.subplots(figsize=(10, 4))
            for var in var_sel:
                info = resultado.get(var, {})

            nombre = info.get("Nombre", var_sel)

            for i in range(1, time_steps + 1):
                if not st.session_state.get("escaneo_activo", True):
                    st.session_state["ultima_figura_anomalias"] = fig_sim
                    break

                ax_sim.clear()
                ax_sim.plot(muestra[:i] / 100, color="green", marker='o', label=nombre)
                ax_sim.set_title(f"🔁 Día {n+1} - {nombre}", fontsize=12, fontweight="bold")
                ax_sim.set_ylabel(f"{nombre} [{info.get('Unidad', '')}]", fontsize=10)
                ax_sim.tick_params(axis='both', labelsize=9)
                ax_sim.grid(True, alpha=0.3)
                ax_sim.set_ylim([min(muestra)/100 - 1, max(muestra)/100 + 1])

                # Líneas de referencia
                if info.get("Valor mínimo") is not None:
                    ax_sim.axhline(info["Valor mínimo"], color='orange', linestyle='--', linewidth=1.2, alpha=0.7,
                                   label=f'Mínimo ({info["Valor mínimo"]:.0f})')
                if info.get("Valor nominal") is not None:
                    ax_sim.axhline(info["Valor nominal"], color='yellow', linestyle='--', linewidth=1.2, alpha=0.7,
                                   label=f'Nominal ({info["Valor nominal"]:.0f})')
                if info.get("Valor máximo") is not None:
                    ax_sim.axhline(info["Valor máximo"], color='orange', linestyle='--', linewidth=1.2, alpha=0.7,
                                   label=f'Máximo ({info["Valor máximo"]:.0f})')

                ax_sim.legend(fontsize=8, loc='upper right')
                contenedor_sim.pyplot(fig_sim)
                time.sleep(velocidad)

            # === Calcular error ===
            secuencia = np.expand_dims(muestra_scaled, axis=0)  # (1, 30, 1)
            pred = model.predict(secuencia)
            error = np.mean(np.square(secuencia - pred), axis=(1, 2))[0]
            health_index.append(error)

            # === Gráfica de health index (solo puntos) ===
            from matplotlib.ticker import MaxNLocator
            # === Gráfica de health index (solo puntos azules y umbral rojo) ===
            fig_hi, ax_hi = plt.subplots(figsize=(8, 3))

            x_vals = list(range(1, len(health_index) + 1))  # Eje X desde 1
            ax_hi.scatter(x_vals, health_index, color="blue", marker='o', label="Health Index")
            ax_hi.axhline(umbral, color="red", linestyle='--', linewidth=1.5, label=f"Umbral ({umbral:.0f})")

            ax_hi.set_title("📉 Health Index", fontsize=12, fontweight="bold")
            ax_hi.set_xlabel("Día", fontsize=10)
            ax_hi.set_ylabel("Health Index", fontsize=10)
            ax_hi.grid(True, alpha=0.3)
            ax_hi.legend(fontsize=8)

            # Eje X con solo números enteros
            ax_hi.xaxis.set_major_locator(MaxNLocator(integer=True))

            contenedor_health.pyplot(fig_hi)
            # Guardar las figuras actuales para mantenerlas tras detener
            st.session_state["ultima_fig_sim"] = fig_sim
            st.session_state["ultima_fig_hi"] = fig_hi


        detener_placeholder.empty()
        st.session_state["escaneo_activo"] = False