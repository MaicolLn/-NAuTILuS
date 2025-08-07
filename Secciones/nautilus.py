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
def calcular_intersecciones_promedio_individual(valores_hi, ventanas, umbral):
#     """
#     Calcula el promedio de días faltantes hasta alcanzar el umbral,
#     proyectando la intersección a partir de una regresión lineal sobre ventanas recientes.

#     Args:
#         valores_hi (list or array): Valores del índice de salud.
#         ventanas (list of int): Tamaños de ventana para ajustar la regresión.
#         umbral (float): Umbral para proyectar la intersección.   

    interceptos = []
    fig_rul, axs = plt.subplots(2, 2, figsize=(16, 10))
    axs = axs.flatten()

    dias_validos = list(range(1, len(valores_hi) + 1))  # días desde 1 a n
    max_maes = valores_hi # MAE máximo por día

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
        if x_interseccion is not None:
            x_max = max(x_interseccion, x_max)
        x_pred = np.linspace(dias_validos[-ventana], x_max, 200)
        y_pred = m * x_pred + b


        if x_interseccion and m > 0:
            faltan_dias = x_interseccion - len(valores_hi)

            if faltan_dias > 1:
                interceptos.append(faltan_dias)

    if interceptos:
        return round(np.mean(interceptos), 1)
    else:
        return None
# def calcular_intersecciones_promedio_individual(valores_hi, ventanas, umbral):
#     """
#     Calcula el promedio de días faltantes hasta alcanzar el umbral,
#     proyectando la intersección a partir de una regresión lineal sobre ventanas recientes.

#     Args:
#         valores_hi (list or array): Valores del índice de salud.
#         ventanas (list of int): Tamaños de ventana para ajustar la regresión.
#         umbral (float): Umbral para proyectar la intersección.

#     Returns:
#         float or None: Días promedio hasta alcanzar el umbral, o None si no hay datos válidos.
#     """
#     from sklearn.linear_model import LinearRegression
#     import numpy as np

#     health_index = np.array(valores_hi)
#     dias_validos = np.arange(1, len(health_index) + 1)
#     interceptos = []

#     for ventana in ventanas:
#         if len(health_index) < ventana:
#             continue

#         X = dias_validos[-ventana:].reshape(-1, 1)
#         y = health_index[-ventana:]

#         modelo = LinearRegression()
#         modelo.fit(X, y)
#         m = modelo.coef_[0]
#         b = modelo.intercept_

#         if m != 0:
#             x_interseccion = (umbral - b) / m
#             faltan_dias = x_interseccion - len(valores_hi)

#             if m > 0 and faltan_dias > 0:
#                 interceptos.append(faltan_dias)

#     if interceptos:
#         return round(np.mean(interceptos), 1)
#     else:
#         return None

def calcular_health_index_subsistema(nombre_subsistema,subsistemas, df_base, modelos, scalers, time_steps):
    if nombre_subsistema in modelos:
        modelo = modelos[nombre_subsistema]
        scaler = scalers[nombre_subsistema]
        variables_disponibles = list(subsistemas[nombre_subsistema])


        muestra_df = df_base[variables_disponibles].sample(n=time_steps, random_state=None).reset_index(drop=True)
        

        vectores = {}
        for var in variables_disponibles:
            vectores[var] = muestra_df[var].values.reshape(-1, 1)

        vector_multivariable = np.concatenate([vectores[var] for var in variables_disponibles], axis=1)
        scaled_vector = scaler.transform(vector_multivariable)
        sequence = np.expand_dims(scaled_vector, axis=0)

        x_pred = modelo.predict(sequence, verbose=0)
        mae_day = np.mean(np.abs(sequence - x_pred))
        return mae_day
    else:
        return None


def nautilus_en_marcha():
    st.header("🚀Health index & RUL")
     
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
        "Sistema de Refrigeración": 0.4,
        "Sistema de Combustible": 0.36,
        "Sistema de Lubricación": 0.5,
        "Temperatura de Gases de Escape": 0.4
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
    iteraciones = st.sidebar.number_input("📅 Días de operación", min_value=2, max_value=60, value=7, step=1)
    velocidad = 0.1
    with st.sidebar.expander("📆 Ventanas de proyección RUL (en días)"):
        try:
            ventana_1 = st.number_input("Ventana 1", min_value=1, value=7, step=1)
            ventana_2 = st.number_input("Ventana 2", min_value=1, value=15, step=1)
            ventana_3 = st.number_input("Ventana 3", min_value=1, value=20, step=1)
            ventana_4 = st.number_input("Ventana 4", min_value=1, value=30, step=1)

            # Juntar en lista ordenada y sin duplicados
            ventanas = sorted(set([ventana_1, ventana_2, ventana_3, ventana_4]))
        except Exception as e:
            st.warning(f"❌ Error en las ventanas: {e}. Usando valores por defecto.")
            ventanas = [7, 15, 30, 60]

    # Crear contenedores si no existen
    if "contenedor_sim" not in st.session_state:
        st.session_state["contenedor_sim"] = st.empty()
    if "contenedor_health" not in st.session_state:
        st.session_state["contenedor_health"] = st.empty()
    if "contenedor_rul" not in st.session_state:
        st.session_state["contenedor_rul"] = st.empty()
    if "contenedor_rul_mensaje" not in st.session_state:
        st.session_state["contenedor_rul_mensaje"] = st.empty()

    if "contenedor_rul_plot" not in st.session_state:
        st.session_state["contenedor_rul_plot"] = st.empty()



    iniciar = st.sidebar.button("▶️ Iniciar simulación")
    detener_placeholder = st.sidebar.empty()
    detener = detener_placeholder.button("⏹️ Detener")

    if detener:
        st.session_state["escaneo_activo"] = False

    # Mostrar último resultado si se detuvo
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
                # Mostrar la figura si existe
        if "ultima_fig_rul_plot" in st.session_state:
            st.session_state["contenedor_rul_plot"].pyplot(st.session_state["ultima_fig_rul_plot"])


    # Lista de subsistemas con modelo asociado
    subsistemas_modelados = {
        "Sistema de Combustible": "combustible",
        "Temperatura de Gases de Escape": "gases",
        "Sistema de Lubricación": "lubricante",
        "Sistema de Refrigeración": "refrigeracion"
        
    }

    

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


    # SIMULACIÓN
    if iniciar:
        st.session_state["escaneo_activo"] = True
        st.session_state["contenedor_sim"].empty()
        st.session_state["contenedor_health"].empty()
        st.session_state["contenedor_rul"].empty()
        st.session_state["contenedor_rul_mensaje"].empty()
        st.session_state["contenedor_rul_plot"].empty()
        
        st.session_state.pop("ultima_fig_sim", None)
        st.session_state.pop("ultima_fig_hi", None)
        st.session_state.pop("ultima_fig_rul", None)
        st.session_state.pop("ultima_fig_rul_plot", None)


        health_index=st.session_state["health_index"][subsistema_sel]
        # Crear columnas para colocar las variables del subsistema
        columnas = st.columns(2)
        contenedor_subsistema = columnas[0].container()  # solo 1 contenedor para todas sus variables

        # Semilla para reproducibilidad
        np.random.seed(42)
        time_steps = 30  # pasos por día
        contenedor_sim = st.session_state["contenedor_sim"]
        contenedor_health = st.session_state["contenedor_health"]
        contenedor_rul = st.session_state["contenedor_rul"]
        
        for dia in range(iteraciones):  # días de operación
            # Crear un diccionario con los vectores reshaped para cada variable
            vectores = {}

            muestra_df = df[var_sel].sample(n=time_steps, random_state=None).reset_index(drop=True)

            vectores = {}
            for var in variables_disponibles:
                vectores[var] = muestra_df[var].values.reshape(-1, 1)

           
            # Concatenar todos los vectores en una sola matriz para análisis multivariable
            vector_multivariable = np.concatenate([vectores[var] for var in variables_disponibles], axis=1)
            scaled_vector = scaler.transform(vector_multivariable)
            sequence = np.expand_dims(scaled_vector, axis=0)  # [1, time_steps, n_features]

            # === 4. Predicción ===
            x_pred = modelo.predict(sequence, verbose=0)

            # === 5. Calcular el error absoluto máximo por día ===
            
            mae_por_variable = np.mean(np.abs(sequence - x_pred), axis=(0, 1))
            mae_day = float(np.mean(mae_por_variable))

            # Guarda el error
            # Guarda el error
            dic_mae_day = {}
            for i, var in enumerate(variables_disponibles):
                mae_day_v = mae_por_variable[i]
                dic_mae_day[var] = mae_day_v

            time.sleep(10)
            reconstrucciones = []
            errores_mse = []
            

            for i in range(1, time_steps + 1):
                if not st.session_state.get("escaneo_activo", True):
                    break

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
                    "axes.titlesize": 16,  # Título del eje
                    "axes.labelsize": 20,  # Etiquetas de ejes
                    "xtick.labelsize": 13,  # Ticks del eje x
                    "ytick.labelsize": 13,  # Ticks del eje y
                    "legend.fontsize": 12,  # Leyenda
                    "figure.titlesize": 24,  # Título general
                })

                n_vars = len(var_sel)
                n_cols = 2
                n_rows = math.ceil(n_vars / n_cols)
                fig_sim, axs = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=False)
                fig_sim.patch.set_facecolor(fondo)
                axs = axs.flatten()
                for ax, var in zip(axs, var_sel):
                    muestra = muestra_df[var].values[:i]
                    info = resultado.get(var, {})
                    hi_var = dic_mae_day[var] 
                    nombre = info.get("Nombre", var)

                    unidad = info.get("Unidad", "")
                    
                    ax.set_title(nombre, fontweight="bold")
                    ax.set_facecolor(fondo)
                    if hi_var is not None:
                        nombre_legenda = f"Health Index {var} | {hi_var:.2f} "
                    else:
                        nombre_legenda = f"Health Index {var} | --- "
                    ax.plot(muestra, color="green", marker='o', label=nombre_legenda)


                    ax.grid(True, alpha=0.3)
                    ax.set_ylim([np.min(muestra) - 1, np.max(muestra) + 1])

                    if info.get("Valor mínimo") is not None:
                        ax.axhline(info["Valor mínimo"], color='orange', linestyle='--', linewidth=1.2, alpha=0.7,
                                label=f"Mínimo ({info['Valor mínimo']})")
                    if info.get("Valor nominal") is not None:
                        ax.axhline(info["Valor nominal"], color='yellow', linestyle='--', linewidth=1.2, alpha=0.7,
                                label=f"Nominal ({info['Valor nominal']})")

                    ax.legend(loc='upper right')
                fig_sim.suptitle(
                    f"🔧 Subsistema: {subsistema_sel} | Día {len(health_index)+1}",
                    fontweight="bold",
                    y=1.4 # ⬆️ súbelo un poco (default es ~0.95)
                )
                contenedor_sim.pyplot(fig_sim)

                # Guarda última figura en sesión
                st.session_state["ultima_fig_sim"] = fig_sim
                
                
                st.session_state["health_index"][subsistema_sel]= health_index

                time.sleep(velocidad)


            # if tipo_datos == "🔴 Datos con anomalías":
            #     sumaa= random.uniform(2, 5)
            #     mae_day= mae_day+sumaa

            st.session_state["health_index"][subsistema_sel].append(mae_day)



            # === 6. Graficar el Health Index ===
            # ✅ Graficar índice de salud acumulado
            fig_hi, ax_hi = plt.subplots(figsize=(8, 4))
            dias = list(range(1, len(health_index) + 1))
            ax_hi.scatter(dias, health_index, marker='o', linestyle='-', color='blue', label=f'Índice de salud día {len(health_index)}: {(mae_day):.2f}')
            ax_hi.axhline(umbral, color="red", linestyle='--', linewidth=1.5, label=f"Umbral ({umbral:.0f})")
            ax_hi.set_title(f"📉 Health Index - Subsistema: {subsistema_sel}")
            ax_hi.set_xlabel("Día")
            ax_hi.set_ylabel("Health Index")
            ax_hi.grid(True, alpha=0.3)
            ax_hi.legend()
            contenedor_health.pyplot(fig_hi)
            st.session_state["ultima_fig_hi"] = fig_hi
            st.session_state["contenedor_health"].pyplot(fig_hi)

            # === RUL Prediction ===
            # ventanas = [7, 15, 30, 60]

            # ssss=calcular_intersecciones_promedio_individual(health_index, ventanas, umbral)
            # st.write(f"calculo de instersección: {ssss}")
            intersecciones = []
            fig_rul, axs = plt.subplots(2, 2, figsize=(16, 10))
            axs = axs.flatten()

            dias = list(range(1, len(health_index) + 1))  # Días desde 1 hasta n
            valores_salud = health_index  # Índice de salud diario (por ejemplo, MAE)

            for i, ventana in enumerate(ventanas):
                if len(dias) < ventana:
                    axs[i].set_visible(False)
                    continue

                # Selección de la ventana
                dias_ventana = np.array(dias[-ventana:])
                salud_ventana = np.array(valores_salud[-ventana:])

                # Suavizado con mediana móvil
                salud_suavizada = pd.Series(salud_ventana).rolling(window=3, center=True).median().dropna().values
                dias_suavizados = dias_ventana[1:-1].reshape(-1, 1)  # Alinear con salud_suavizada

                if len(salud_suavizada) <= 1:
                    axs[i].set_visible(False)
                    continue

                # Ajuste lineal con sklearn
                regresion = LinearRegression()
                regresion.fit(dias_suavizados, salud_suavizada)

                pendiente = regresion.coef_[0]
                intercepto = regresion.intercept_

                # Cálculo de intersección con el umbral
                dia_interseccion = None
                if pendiente != 0:
                    dia_interseccion = (umbral - intercepto) / pendiente

                # Rango de proyección de la línea
                dia_inicio = dias_suavizados[0][0]
                dia_max = dias[-1] + 10
                if dia_interseccion and pendiente > 0:
                    dia_max = max(dia_max, dia_interseccion)

                dias_prediccion = np.linspace(dia_inicio, dia_max, 200).reshape(-1, 1)
                salud_predicha = regresion.predict(dias_prediccion)

                # Graficar
                ax = axs[i]
                ax.scatter(dias, valores_salud, color='blue', s=60, label="Health Index")
                ax.plot(dias_prediccion, salud_predicha, color='red', label='Tendencia suavizada')
                ax.axhline(umbral, color='red', linestyle='dotted', linewidth=2, label='Umbral')

                titulo = f"Ventana: {ventana} días"

                if dia_interseccion and pendiente > 0:
                    dias_restantes = dia_interseccion - len(valores_salud)
                    ax.scatter(dia_interseccion, umbral, color='black', s=40, zorder=5)

                    if dias_restantes > 1:
                        ax.text(
                            dia_interseccion + 1, umbral, f"Faltan {dias_restantes:.1f} días",
                            bbox=dict(facecolor=fondo, edgecolor=color_letra)
                        )
                    else:
                        titulo += " 🔴"

                    intersecciones.append(dia_interseccion)

                

                ax.set_title(titulo)
                ax.set_xlabel("Días")
                ax.set_ylabel("Índice de Salud")
                ax.legend()


            fig_rul.suptitle(f"🔮 Proyección de RUL tras día {len(health_index)}", fontweight='bold', y=1.02)
            plt.tight_layout()
            contenedor_rul.pyplot(fig_rul)
            # Mostrar gráfico en contenedor
            st.session_state["ultima_fig_sim"] = fig_sim
            st.session_state["ultima_fig_hi"] = fig_hi
            st.session_state["ultima_fig_rul"] = fig_rul


            st.session_state["health_index"][subsistema_sel] = health_index



            interceptos=intersecciones
            # === Mostrar mensaje RUL estimado si hay intersecciones válidas ===
            if interceptos:
                intercepción_hi = float(np.mean(interceptos))
                rul_promedio= intercepción_hi- len(health_index)
                st.session_state["remaining_useful_life"][subsistema_sel].append(rul_promedio)
                if intercepción_hi>( len(health_index) + 1):
                    st.session_state["ultimo_rul_mensaje"] = f"""<div style='
                            background-color:#f0f2f6;
                            padding: 10px 15px;
                            border-left: 5px solid #6c63ff;
                            border-radius: 6px;
                            font-size: 16px;
                            font-weight: bold;
                            color: #333;'>
                        📌 <span style='color:#6c63ff;'>RUL estimado:</span> {(rul_promedio):.1f} días para superar el health index permitido.
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
                    📌 <span style='color:#6c63ff;'>RUL estimado: Umbral superado - Día {intercepción_hi:.0f} </span> Revisión urgente, se ha superado el health index límite.
                    </div>"""
                    st.session_state["contenedor_rul_mensaje"].markdown(
                        st.session_state["ultimo_rul_mensaje"], unsafe_allow_html=True
                    )
            # Validar que haya datos antes de graficar
            else: 
                st.session_state["remaining_useful_life"][subsistema_sel].append(st.session_state["remaining_useful_life"][subsistema_sel][-1])
                st.session_state["ultimo_rul_mensaje"] = f"""<div style='
                        background-color:#f0f2f6;
                        padding: 10px 15px;
                        border-left: 5px solid #6c63ff;
                        border-radius: 6px;
                        font-size: 16px;
                        font-weight: bold;
                        color: #333;'>
                    📌 <span style='color:#6c63ff;'>RUL estimado:</span> {(st.session_state["remaining_useful_life"][subsistema_sel][-1]):.1f} días para superar el health index permitido.
                </div>"""
                st.session_state["contenedor_rul_mensaje"].markdown(
                    st.session_state["ultimo_rul_mensaje"], unsafe_allow_html=True
                    )
            rul_list_plot = st.session_state["remaining_useful_life"].get(subsistema_sel, [])
            if rul_list_plot:
                fig_rul_plot, ax_rul_plot = plt.subplots(figsize=(10, 5))

                dias = list(range(1, len(rul_list_plot) + 1))

                # Estética mejorada
                ax_rul_plot.plot(
                    dias, rul_list_plot,
                    marker='o',
                    linestyle='-',
                    linewidth=2,
                    markersize=6,
                    color='#007acc',
                    label=f'📉 RUL: {rul_list_plot[-1]:.0f} Días'
                )

                # Estética general
                ax_rul_plot.set_title(
                    f"🔧 Remaining Useful Life - {subsistema_sel}",
                    fontsize=14, fontweight='bold', color=color_letra
                )
                ax_rul_plot.set_xlabel("Día", fontsize=12)
                ax_rul_plot.set_ylabel("RUL estimado (Días)", fontsize=12)
                ax_rul_plot.tick_params(axis='both', labelsize=10)
                ax_rul_plot.grid(True, linestyle='--', alpha=0.4)

                # Umbral opcional
                # umbral_rul = 50
                # ax_rul_plot.axhline(umbral_rul, color='red', linestyle='--', linewidth=1.5, label=f'Umbral ({umbral_rul})')

                ax_rul_plot.legend(fontsize=10, loc='upper right', frameon=False)

                # Quitar borde del gráfico
                for spine in ['top', 'right']:
                    ax_rul_plot.spines[spine].set_visible(False)

                # Mostrar en contenedor y guardar en session_state
                st.session_state["contenedor_rul_plot"].pyplot(fig_rul_plot)
                st.session_state["ultima_fig_rul_plot"] = fig_rul_plot
            else:
                st.session_state["contenedor_rul_plot"].warning("No hay datos de RUL para graficar.")


        detener_placeholder.empty()
        st.session_state["escaneo_activo"] = False

