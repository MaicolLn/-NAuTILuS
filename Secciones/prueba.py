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
from sklearn.linear_model import RANSACRegressor, LinearRegression
import plotly.graph_objects as go

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


def nautilus_en_marcha_2():
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
        "Sistema de Refrigeración": 0.25,
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
            ventana= st.number_input("Ventana 1", min_value=1, value=30, step=1)
            # ventana_2 = st.number_input("Ventana 2", min_value=1, value=30, step=1)
            # ventana_3 = st.number_input("Ventana 3", min_value=1, value=60, step=1)
            # ventana_4 = st.number_input("Ventana 4", min_value=1, value=120, step=1)

            # # Juntar en lista ordenada y sin duplicados
            # ventanas = sorted(set([ventana_1, ventana_2, ventana_3, ventana_4]))
        except Exception as e:
            st.warning(f"❌ Error en las ventanas: {e}. Usando valores por defecto.")
            ventana= 30

    # Crear contenedores si no existen
    if "contenedor_sim" not in st.session_state:
        st.session_state["contenedor_sim"] = st.empty()
    if "contenedor_health" not in st.session_state:
        st.session_state["contenedor_health"] = st.empty()
    if "contenedor_rul" not in st.session_state:
        st.session_state["contenedor_rul"] = st.empty()
    if "contenedor_rul_plot" not in st.session_state:
        st.session_state["contenedor_rul_plot"] = st.empty()
    if "contenedor_rul_mensaje" not in st.session_state:
        st.session_state["contenedor_rul_mensaje"] = st.empty()
    if "contenedor_rul_resumen" not in st.session_state:
        st.session_state["contenedor_rul_resumen"] = st.empty()
    if "contenedor_rul_resumen_var" not in st.session_state:
        st.session_state["contenedor_rul_resumen_var"] = st.empty()




    iniciar = st.sidebar.button("▶️ Iniciar simulación")
    detener_placeholder = st.sidebar.empty()
    detener = detener_placeholder.button("⏹️ Detener")

    if detener:
        st.session_state["escaneo_activo"] = False

    # Mostrar último resultado si se detuvo
    if not st.session_state.get("escaneo_activo", True):
        if "ultima_fig_sim" in st.session_state:
            st.session_state["contenedor_sim"].pyplot(st.session_state["ultima_fig_sim"])
        # if "ultima_fig_hi" in st.session_state:
        #     st.session_state["contenedor_health"].pyplot(st.session_state["ultima_fig_hi"])
        if "ultima_fig_rul" in st.session_state and "contenedor_rul" in st.session_state:
            fig = st.session_state["ultima_fig_rul"]
            contenedor = st.session_state["contenedor_rul"]
            contenedor.plotly_chart(fig, use_container_width=True)
                            # Mostrar la figura si existe
        if "ultima_fig_rul_plot" in st.session_state and "contenedor_rul_plot" in st.session_state:
            fig = st.session_state["ultima_fig_rul_plot"]
            contenedor = st.session_state["contenedor_rul_plot"]
            contenedor.plotly_chart(fig, use_container_width=True)

        if "ultimo_rul_mensaje" in st.session_state:
            st.session_state["contenedor_rul_mensaje"].markdown(
                st.session_state["ultimo_rul_mensaje"], unsafe_allow_html=True
            )

        if "ultimo_rul_resumen" in st.session_state:
            st.session_state["contenedor_rul_resumen"].markdown(
                st.session_state["ultimo_rul_resumen"], unsafe_allow_html=True
            )
        if "ultimo_rul_resumen_var" in st.session_state:
            st.session_state["contenedor_rul_resumen_var"].markdown(
                st.session_state["ultimo_rul_resumen_var"], unsafe_allow_html=True
            )
            


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
        st.session_state["contenedor_rul_plot"].empty()
        st.session_state["contenedor_rul_mensaje"].empty()
        st.session_state["contenedor_rul_resumen"].empty()
        st.session_state["contenedor_rul_resumen_var"].empty()

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
        # contenedor_health = st.session_state["contenedor_health"]
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
            import plotly.graph_objects as go

            # === RUL Prediction con Plotly ===
            dias_x = list(range(1, len(health_index) + 1))
            valores_y = health_index

            fig_rul = go.Figure()

            # Datos reales
            fig_rul.add_trace(go.Scatter(
                x=dias_x,
                y=valores_y,
                mode='markers',
                name=f'Health Index día {len(health_index)}: {mae_day:.2f}',
                marker=dict(color='blue'),
                hovertemplate='Día: %{x}<br>Health Index: %{y:.2f}<extra></extra>'
            ))

            # # Línea de umbral
            # fig_rul.add_trace(go.Scatter(
            #     x=[min(dias_x), max(x_proyeccion)],
            #     y=[umbral, umbral],
            #     mode='lines',
            #     name=f'Umbral: {umbral}',
            #     line=dict(color='red', dash='dot'),
            #     hoverinfo='skip'  # No mostrar tooltip para esta línea
            # ))

            # Ajuste lineal y predicción
            if len(dias_x) >= ventana:
                dias_modelo = np.array(dias_x[-ventana:])
                salud_modelo = np.array(valores_y[-ventana:])

                salud_suavizada = pd.Series(salud_modelo).rolling(window=3, center=True).median().dropna().values
                dias_suavizados = dias_modelo[1:-1].reshape(-1, 1)

                if len(salud_suavizada) > 1:
                    modelo_lineal = LinearRegression()
                    modelo_lineal.fit(dias_suavizados, salud_suavizada)

                    pendiente = modelo_lineal.coef_[0]
                    intercepto = modelo_lineal.intercept_

                    x_interseccion = (umbral - intercepto) / pendiente if pendiente != 0 else None
                    x_inicio = dias_suavizados[0][0]

                    if (x_interseccion is not None and x_interseccion > len(health_index)) and pendiente > 0:
                        x_proyeccion = np.linspace(x_inicio, x_interseccion, 200)
                    else:
                        x_proyeccion = np.linspace(x_inicio, dias_x[-1] + 10, 200)

                    y_proyeccion = modelo_lineal.predict(x_proyeccion.reshape(-1, 1))
                    



                    # Punto de intersección
                    if x_interseccion is not None and pendiente > 0:
                        dias_restantes = x_interseccion - len(health_index)
                        interceptos_ventana = x_interseccion 
                        # === Datos reales sin conectar ===
                    else:
                        interceptos_ventana = None

            fig_rul = go.Figure()
            # luego, agregar las trazas

            fig_rul.add_trace(go.Scatter(
                x=dias_x,
                y=valores_y,
                mode='markers',
                name=f'Health Index día {len(health_index)}: {mae_day:.2f}',
                marker=dict(color='blue'),
                hovertemplate='Día: %{x}<br>Health Index: %{y:.2f}<extra></extra>'
            ))

            # Punto negro en la intersección (si existe)
            if x_interseccion is not None and pendiente > 0:
                fig_rul.add_trace(go.Scatter(
                    x=[x_interseccion],
                    y=[umbral],
                    mode='markers',
                    name=f'Intersección estimada el día: {x_interseccion:.0f}',
                    marker=dict(color='black', size=10, symbol='circle'),
                    hovertemplate='Día estimado: %{x:.1f}<br>Health Index: %{y:.2f}<extra></extra>'
                ))


            # === Línea de umbral extendida ===
            x_umbral_min = min(dias_x)
            x_umbral_max = max(x_proyeccion) + 5 if len(health_index)< x_interseccion else max(dias_x)

            # Línea de tendencia
            fig_rul.add_trace(go.Scatter(
                x=x_proyeccion,
                y=y_proyeccion,
                mode='lines',
                name='Tendencia',
                line=dict(color='red'),
                hoverinfo='skip'
            ))

            fig_rul.add_trace(go.Scatter(
                 x=[0, max(x_proyeccion) + 5],
                y=[umbral] * len(x_proyeccion),
                mode='lines',
                name=f'Umbral: {umbral}',
                line=dict(color='red', dash='dot'),
                hoverinfo='skip'
            ))

            # Layout general
            fig_rul.update_layout(
                title=f"🔮 Proyección de RUL tras día {len(health_index)}",
                xaxis_title="Día",
                yaxis_title="Health Index",
                height=600,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

            # Mostrar en Streamlit
            contenedor_rul.plotly_chart(fig_rul, use_container_width=True)



            # Mostrar gráfico en contenedor
            st.session_state["ultima_fig_sim"] = fig_sim
            # st.session_state["ultima_fig_hi"] = fig_hi
            st.session_state["ultima_fig_rul"] = fig_rul


            st.session_state["health_index"][subsistema_sel] = health_index



            # === Mostrar mensaje RUL estimado si hay intersecciones válidas ===
            if interceptos_ventana:
                intercepción_hi = interceptos_ventana 
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
                        📌 <span style='color:#6c63ff;'>RUL estimado - día {len(health_index):.0f}:</span> Aproximadamente {(rul_promedio):.0f} días para superar el Health Index permitido.
                    </div>"""
                    # st.session_state["contenedor_rul_mensaje"].markdown(
                    #     st.session_state["ultimo_rul_mensaje"], unsafe_allow_html=True
                    # )
                else:
                    st.session_state["ultimo_rul_mensaje"] = f"""<div style='
                        background-color:#f0f2f6;
                        padding: 10px 15px;
                        border-left: 5px solid #6c63ff;
                        border-radius: 6px;
                        font-size: 16px;
                        font-weight: bold;
                        color: #333;'>
                        📌 <span style='color:#6c63ff;'>Adevertencia:</span> El límite del Health Index se sobrepasó hace {abs((rul_promedio)):.0f} días.
                    </div>"""
                    
                    # st.session_state["contenedor_rul_mensaje"].markdown(
                    #     st.session_state["ultimo_rul_mensaje"], unsafe_allow_html=True
                    # )
            # Validar que haya datos antes de graficar
            else: 
                st.session_state["remaining_useful_life"][subsistema_sel].append(None)

                st.session_state["ultimo_rul_mensaje"] = f"""<div style='
                        background-color:#f0f2f6;
                        padding: 10px 15px;
                        border-left: 5px solid #6c63ff;
                        border-radius: 6px;
                        font-size: 16px;
                        font-weight: bold;
                        color: #333;'>
                    📌 <span style='color:#6c63ff;'>RUL estimado - día {len(health_index):.0f}:</span> No se ha estimado algún comportamiento que indique superar el umbral del Health Index.
                </div>"""
                # st.session_state["contenedor_rul_mensaje"].markdown(
                #     st.session_state["ultimo_rul_mensaje"], unsafe_allow_html=True
                #     )
            rul_list_plot = st.session_state["remaining_useful_life"].get(subsistema_sel, [])

            if rul_list_plot:
                dias = list(range(32, 30 + len(rul_list_plot)))  # ej. si hay 10 datos → [31, ..., 40]
                fig_rul_plot = go.Figure()

                fig_rul_plot.add_trace(go.Scatter(
                    x=dias,
                    y=rul_list_plot,
                    mode='lines+markers',
                    marker=dict(size=8, color='#007acc'),
                    name=f'📉 RUL: {rul_list_plot[-1]:.0f} Días' if rul_list_plot[-1] is not None else '📉 RUL no estimado',
                    hovertemplate='Día %{x}<br>RUL %{y:.0f} días<extra></extra>'
                ))


                # Opcional: línea de umbral (comentado si no la usas)
                # umbral_rul = 50
                # fig_rul_plot.add_hline(y=umbral_rul, line_dash="dash", line_color="red", annotation_text=f"Umbral {umbral_rul}")

                fig_rul_plot.update_layout(
                    title=f"🔧 Remaining Useful Life - {subsistema_sel}",
                    xaxis_title="Día",
                    yaxis_title="RUL estimado (Días)",
                    yaxis=dict(autorange=True),
                    xaxis=dict(
                        tickmode='linear',
                        tick0=32,
                        dtick=1,
                        range=[32, 5+ len(health_index)],
                    ),
                    template="simple_white",
                    font=dict(size=12, color=color_letra),
                    height=400,
                    legend=dict(font=dict(size=10), x=0.01, y=0.99),
                    margin=dict(l=40, r=20, t=50, b=40),
                )


                st.session_state["contenedor_rul_plot"].plotly_chart(fig_rul_plot, use_container_width=True)
                st.session_state["ultima_fig_rul_plot"] = fig_rul_plot

            else:
                st.session_state["contenedor_rul_plot"].warning("No hay datos de RUL para graficar.")

            # Supongamos que estás dentro de un bucle sobre las variables seleccionadas
            # 🧩 Resumen general del día (antes del resumen por variable)
            dia_actual = len(health_index)
            hi_day = mae_day  # Asegúrate de que está definido
            subsistema = subsistema_sel  # Asegúrate de que está definido

            hi_formateado_general = f"{hi_day:.2f}" if hi_day is not None else "---"

            resumen_general_html = f"""
            <div style='
                background-color:#f0f2f6;
                padding: 10px 15px;
                border-left: 5px solid #6c63ff;
                border-radius: 6px;
                font-size: 16px;
                color: #333;
                margin-bottom: 8px;'>
                📅 <span style='font-weight:bold; color:#6c63ff;'>Día</span>: <span style='font-weight:bold;'>{dia_actual}</span> |
                ⚙️ <span style='font-weight:bold;'>{subsistema}</span> |
                📈 <span style='font-weight:bold; color:#6c63ff;'>Health Index</span>: <span style='font-weight:bold;'>{hi_formateado_general}</span>
            </div>
            """


            # 🧪 Resumen por variable
            resumen_general_html  # Inicializa con el resumen general
            resumen_html_var= f""
            for var in var_sel:
                info = resultado.get(var, {})
                hi_var = dic_mae_day.get(var, None)
                nombre = info.get("Nombre", var)
                hi_formateado = f"{hi_var:.2f}" if hi_var is not None else "---"

                resumen_html_var += f"""
                <div style='
                    background-color:#f0f2f6;
                    padding: 10px 15px;
                    border-left: 5px solid #6c63ff;
                    border-radius: 6px;
                    font-size: 16px;
                    font-weight: bold;
                    color: #333;
                    margin-bottom: 8px;'>
                    🧪 <span style='color:#6c63ff;'>Variable:</span> {nombre} &nbsp;|&nbsp;
                    ❤️ <span style='color:#6c63ff;'>Health Index:</span> {hi_formateado}
                </div>
                """


            # Al final del bucle o luego de construir todo el resumen
            # st.session_state["contenedor_rul_mensaje"].markdown(
            #     st.session_state["ultimo_rul_mensaje"], unsafe_allow_html=True
            #     )


            st.session_state["ultimo_rul_resumen"] = resumen_general_html
            st.session_state["contenedor_rul_resumen"].markdown(
                st.session_state["ultimo_rul_resumen"], unsafe_allow_html=True
            )



            st.session_state["ultimo_rul_resumen_var"] = resumen_html_var
            st.session_state["contenedor_rul_resumen_var"].markdown(
                st.session_state["ultimo_rul_resumen_var"], unsafe_allow_html=True
            )


            st.session_state["contenedor_rul_mensaje"].markdown(
                st.session_state["ultimo_rul_mensaje"], unsafe_allow_html=True
                )
        detener_placeholder.empty()
        st.session_state["escaneo_activo"] = False

