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



def PanelC():
    st.header("🚢 Nautilus en marcha")
    
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("colorblind")

    df_1 = st.session_state.get("datos_modelo_1")
    df_2 = st.session_state.get("datos_modelo_2")
    df_3 = st.session_state.get("datos_modelo_3")
    resultado = st.session_state.get("resultado")
    subsistemas = st.session_state.get("subsistemas")

    if not resultado or (df_1 is None and df_2 is None):
        st.warning("⚠️ Asegúrate de haber generado datos y cargado el diccionario `resultado`.")
        st.stop()

    st.sidebar.subheader("⚙️ Simulación")
    modelo = st.sidebar.selectbox(
        "🟦 Tipo de datos a visualizar",
        ["🟢 Datos sin anomalías", "🔴 Datos con anomalías", "🔵 Datos de operación "]
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

    if not variables_disponibles:
        st.warning("No hay variables válidas para este subsistema.")
        st.stop()

    st.sidebar.markdown("📌 Variables del subsistema:")
    var_sel = []
    for var in variables_disponibles:
        if st.sidebar.checkbox(var, value=True):
            var_sel.append(var)
    iteraciones = st.sidebar.number_input("📅 Días de operación", min_value=2, max_value=60, value=7, step=1)
    velocidad = st.sidebar.slider("Velocidad de simulación", 0.01, 2.0, 0.5, 0.1)
    with st.sidebar.expander("📆 Ventanas de proyección RUL (en días)"):
        try:
            ventana_1 = st.number_input("Ventana 1", min_value=1, value=7, step=1)
            ventana_2 = st.number_input("Ventana 2", min_value=7, value=30, step=1)
            ventana_3 = st.number_input("Ventana 3", min_value=60, value=60, step=1)
            ventana_4 = st.number_input("Ventana 4", min_value=120, value=120, step=1)

            # Juntar en lista ordenada y sin duplicados
            ventanas = sorted(set([ventana_1, ventana_2, ventana_3, ventana_4]))
        except Exception as e:
            st.warning(f"❌ Error en las ventanas: {e}. Usando valores por defecto.")
            ventanas = [7, 15, 30, 60]
    umbral = 3.0

    # Crear contenedores si no existen
    if "contenedor_sim" not in st.session_state:
        st.session_state["contenedor_sim"] = st.empty()
    if "contenedor_health" not in st.session_state:
        st.session_state["contenedor_health"] = st.empty()
    if "contenedor_rul" not in st.session_state:
        st.session_state["contenedor_rul"] = st.empty()
    if "contenedor_rul_mensaje" not in st.session_state:
        st.session_state["contenedor_rul_mensaje"] = st.empty()

    iniciar = st.sidebar.button("▶️ Iniciar simulación")
    detener_placeholder = st.sidebar.empty()
    detener = detener_placeholder.button("⏹️ Detener")

    if detener:
        st.session_state["escaneo_activo"] = False

    if not st.session_state.get("escaneo_activo", True):
        if "ultima_fig_sim" in st.session_state:
            st.session_state["contenedor_sim"].pyplot(st.session_state["ultima_fig_sim"])
        if "ultima_fig_hi" in st.session_state:
            st.session_state["contenedor_health"].pyplot(st.session_state["ultima_fig_hi"])
        if "ultima_fig_rul" in st.session_state:
            st.session_state["contenedor_rul"].pyplot(st.session_state["ultima_fig_rul"])
        if "ultimo_rul_mensaje" in st.session_state:
            st.session_state["contenedor_rul_mensaje"].markdown(
                st.session_state["ultimo_rul_mensaje"], unsafe_allow_html=True
            )

    modelo_dir = os.path.join(os.getcwd(), "modelos")
    try:
        model = load_model(os.path.join(modelo_dir, "modelo_lstm_vae.h5"), compile=False)
        scaler = joblib.load(os.path.join(modelo_dir, "scaler_lstm_vae.pkl"))
    except Exception as e:
        st.error(f"❌ Error al cargar modelo o scaler: {e}")
        st.stop()

    if iniciar:
        st.session_state["escaneo_activo"] = True
        st.session_state["contenedor_sim"].empty()
        st.session_state["contenedor_health"].empty()
        st.session_state["contenedor_rul"].empty()
        st.session_state["contenedor_rul_mensaje"].empty()
        st.session_state.pop("ultima_fig_sim", None)
        st.session_state.pop("ultima_fig_hi", None)
        st.session_state.pop("ultima_fig_rul", None)


        # Health Index precargado
        health_index= st.session_state["health_index"][subsistema_sel]
        np.random.seed(42)
        time_steps = 30
        contenedor_sim = st.session_state["contenedor_sim"]
        contenedor_health = st.session_state["contenedor_health"]
        contenedor_rul = st.session_state["contenedor_rul"]

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

            
            tema = st.get_option("theme.base")
            fondo = "#FFFFFF00" if tema == "light" else "#B2ACAC00"
            color_letra = "#000000FF" if tema == "light" else "white"

            plt.rcParams.update({
                "axes.facecolor": fondo,
                "figure.facecolor": fondo,
                "axes.labelcolor": color_letra,
                "xtick.color": color_letra,
                "ytick.color": color_letra,
                "text.color": color_letra,
                "axes.edgecolor": color_letra,
                "savefig.facecolor": fondo,
                "legend.labelcolor": color_letra,
                "lines.linewidth": 2.5,
                    
                # ✅ Tamaño de fuente general
                "font.size": 16,  # Fuente base
                "axes.titlesize": 20,  # Título del eje
                "axes.labelsize": 20,  # Etiquetas de ejes
                "xtick.labelsize": 18,  # Ticks del eje x
                "ytick.labelsize": 18,  # Ticks del eje y
                "legend.fontsize": 13,  # Leyenda
                "figure.titlesize": 24,  # Título general

            })
            fig_sim, ax_sim = plt.subplots(figsize=(10, 4))
            fig_sim.patch.set_facecolor(fondo)
            for var in var_sel:
                info = resultado.get(var, {})
            nombre = info.get("Nombre", var_sel)

            for i in range(1, time_steps + 1):
                if not st.session_state.get("escaneo_activo", True):
                    break
                ax_sim.clear()
                ax_sim.plot(muestra[:i] / 100, color="green", marker='o', label=nombre)
                ax_sim.set_title(f"🔁 Día {len(health_index) +1} - {nombre}", fontweight="bold")
                ax_sim.set_ylabel(f"{nombre} [{info.get('Unidad', '')}]", fontsize=10)
                ax_sim.grid(True, alpha=0.3)
                ax_sim.set_ylim([np.min(muestra)/100 - 1, np.max(muestra)/100 + 1])
                if info.get("Valor mínimo") is not None:
                    ax_sim.axhline(
                        info["Valor mínimo"], color='orange', linestyle='--', linewidth=1.2, alpha=0.7,
                        label=f"Mínimo ({info['Valor mínimo']})"
                    )
                if info.get("Valor nominal") is not None:
                    ax_sim.axhline(
                        info["Valor nominal"], color='yellow', linestyle='--', linewidth=1.2, alpha=0.7,
                        label=f"Nominal ({info['Valor nominal']})"
                    )
                if info.get("Valor máximo") is not None:
                    ax_sim.axhline(
                        info["Valor máximo"], color='orange', linestyle='--', linewidth=1.2, alpha=0.7,
                        label=f"Máximo ({info['Valor máximo']})"
                    )

                ax_sim.legend(loc='upper right')
                contenedor_sim.pyplot(fig_sim)
                time.sleep(velocidad)

            secuencia = np.expand_dims(muestra_scaled, axis=0)
            pred = model.predict(secuencia)
            error = np.mean(np.square(secuencia - pred), axis=(1, 2))[0]
            health_index.append(error)

            # === Health Index Plot ===
            fig_hi, ax_hi = plt.subplots(figsize=(8, 3))
            x_vals = list(range(1, len(health_index) + 1))
            ax_hi.scatter(x_vals, health_index, color="blue", marker='o', label=f"Health Index: {error:.1f}")
            ax_hi.axhline(umbral, color="red", linestyle='--', linewidth=1.5, label=f"Umbral ({umbral:.0f})")
            ax_hi.set_title("📉 Health Index", fontsize=12, fontweight="bold")
            ax_hi.set_xlabel("Día", fontsize=10)
            ax_hi.set_ylabel("Health Index", fontsize=10)
            ax_hi.grid(True, alpha=0.3)
            ax_hi.legend(fontsize=8)
            ax_hi.xaxis.set_major_locator(MaxNLocator(integer=True))
            contenedor_health.pyplot(fig_hi)

            # === RUL Prediction ===
            interceptos = []
            fig_rul, axs = plt.subplots(2, 2, figsize=(16, 10))
            axs = axs.flatten()
            dias_validos = list(range(1, len(health_index) + 1))
            max_maes = health_index

            for i, ventana in enumerate(ventanas):
                if len(dias_validos) < ventana:
                    axs[i].set_visible(False)
                    continue

                X_interp = np.array(dias_validos[-ventana:]).reshape(-1, 1)
                y_interp = np.array(max_maes[-ventana:])

                modelo_lin = LinearRegression()
                modelo_lin.fit(X_interp, y_interp)
                m = modelo_lin.coef_[0]
                b = modelo_lin.intercept_
                x_interseccion = None
                if m != 0:
                    x_interseccion = (umbral - b) / m

                x_max = dias_validos[-1] + 10
                if x_interseccion:
                    x_max = max(x_interseccion, x_max)
                x_pred = np.linspace(dias_validos[-ventana], x_max, 200)
                y_pred = m * x_pred + b

                ax = axs[i]
                ax.scatter(dias_validos, max_maes, color='blue', s=60, label="Health Index")
                ax.plot(x_pred, y_pred, color='red', label='Proyección lineal')
                ax.axhline(umbral, color='red', linestyle='dotted', linewidth=2, label='Umbral')

                # etiquetas = {7: "semanal", 15: "quincenal", 30: "mensual", 60: "bimestral"}
                # titulo = f"Ventana: {ventana} días ({etiquetas.get(ventana, '')})"
                
                titulo = f"Ventana: {ventana} días"

                if x_interseccion and m > 0:
                    faltan_dias = x_interseccion - len(health_index)
                    ax.scatter(x_interseccion, umbral, color='black', s=40, zorder=5)

                    if faltan_dias > 1:
                        ax.text(
                            x_interseccion + 1, umbral, f"Faltan {faltan_dias:.1f} días", fontsize=9,
                            bbox=dict(facecolor=fondo, edgecolor=color_letra)
                        )
                    else:
                        # 🔴 Punto rojo en el título
                        titulo += " 🔴"

                    interceptos.append(x_interseccion)

                ax.set_title(titulo)
                ax.set_xlabel("Día")
                ax.set_ylabel("Health Index")
                ax.grid(True)
                ax.legend()

            fig_rul.suptitle(f"Proyección de RUL tras día {len(health_index)}", fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            contenedor_rul.pyplot(fig_rul)

            # Guardar última figura
            # Guardar última figura
            st.session_state["ultima_fig_sim"] = fig_sim
            st.session_state["ultima_fig_hi"] = fig_hi
            st.session_state["ultima_fig_rul"] = fig_rul
            st.session_state["health_index"][subsistema_sel] = health_index
            # Acceder al contenedor del mensaje RUL
            # Mostrar mensaje visualmente atractivo del RUL

            if interceptos:
                promedio_rul = float(np.mean(interceptos))
                if promedio_rul>( len(health_index) + 1):
                    st.session_state["ultimo_rul_mensaje"] = f"""<div style='
                            background-color:#f0f2f6;
                            padding: 10px 15px;
                            border-left: 5px solid #6c63ff;
                            border-radius: 6px;
                            font-size: 16px;
                            font-weight: bold;
                            color: #333;'>
                        📌 <span style='color:#6c63ff;'>RUL estimado:</span> {(promedio_rul - len(health_index)):.1f} días para superar el health index permitido.
                    </div>"""
                    st.session_state["contenedor_rul_mensaje"].markdown(
                        st.session_state["ultimo_rul_mensaje"], unsafe_allow_html=True
                    )
                else:
                    st.session_state["ultimo_rul_mensaje"] = f"""<div style='
                        background-color:#f0f2f6;
                        padding: 10px 15px;
                        border-left: 5px solid #6c63ff;
                        border-radius: 6px;
                        font-size: 16px;
                        font-weight: bold;
                        color: #333;'>
                    📌 <span style='color:#6c63ff;'>RUL estimado: Umbral superado - Día {promedio_rul:.0f} </span> Revisión urgente, se ha superado el health index límite.
                    </div>"""
                    st.session_state["contenedor_rul_mensaje"].markdown(
                        st.session_state["ultimo_rul_mensaje"], unsafe_allow_html=True
                    )
        detener_placeholder.empty()
        st.session_state["escaneo_activo"] = False
