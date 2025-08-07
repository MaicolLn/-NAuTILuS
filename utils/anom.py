import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import math
import streamlit as st

def anomalias(df, modelo, scaler, variables_disponibles, resultado, subsistema_sel, factor_escala=1.0, limites=None):
    st.write(f"🔍 Detección de Anomalías - {subsistema_sel}")

    # Tema Streamlit
    tema = st.get_option("theme.base")
    fondo = "#FFFFFF" if tema == "light" else "#1E1E1E"
    color_letra = "#070707" if tema == "light" else "white"
    color_grid = "#DDDDDD" if tema == "light" else "#44444400"
        # === 2. Umbrales específicos por subsistema ===
    umbrales = {
        "Sistema de Refrigeración": 0.4,
        "Sistema de Combustible": 0.36,
        "Sistema de Lubricación": 0.5,
        "Temperatura de Gases de Escape": 0.4
    }
    # umbral = float(umbrales[subsistema_sel])

    n_datos=30
    # Estilo visual
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
        "grid.color": color_grid,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "lines.linewidth": 2.5,
        "font.size": 16,
        "axes.titlesize": 12,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "figure.titlesize": 24,
    })

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("colorblind")

    # Validar variables
    variables = [v for v in variables_disponibles if v in resultado]
    n_vars = len(variables)

    if n_vars == 0:
        st.warning("No hay variables válidas para graficar.")
        return

    # Crear figura
    n_cols = 2
    n_rows = math.ceil(n_vars / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=False)
    fig.patch.set_facecolor(fondo)
    axs = axs.flatten()
    muestra_df = df


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
    errores_todas = np.abs(sequence[0] - x_pred[0])  # shape: (76, n_features)
    errores_dict = {var: errores_todas[:, idx] for idx, var in enumerate(variables_disponibles)}
    errores_df = pd.DataFrame(errores_dict)


    n = len(variables)
    fig, axs = plt.subplots(n, 2, figsize=(14, 3.5 * n))  # 2 columnas: Real - Error

    for i, var in enumerate(variables):
        info = resultado.get(var, {})
        nombre = info.get("Nombre", var)
        umbral = st.session_state["umbrales"].get(var)


        # Obtener valores reales
        if muestra_df is not None and var in muestra_df.columns:
            valores_1 = muestra_df[var].values / factor_escala
        else:
            continue

        # Obtener errores
        errores = errores_dict.get(var, np.zeros_like(valores_1))

        # === Gráfica 1: Señal real ===
        ax_real = axs[i, 0]
        ax_real.set_facecolor(fondo)
        ax_real.plot(valores_1, color="blue", linestyle="-", alpha=0.8, label="Real")
        ax_real.set_title(f"{nombre}", fontweight="bold", color=color_letra)
        ax_real.tick_params(axis='both', labelsize=9, colors=color_letra)
        ax_real.grid(True, alpha=0.4)

        # Aplicar límites del eje Y si existen
        if limites and var in limites:
            ymin, ymax = limites[var]
            ax_real.set_ylim(ymin, ymax)

        if len(valores_1) > 0:
            indices_fuera_limite = []

            # Líneas de referencia
            if info.get("Valor mínimo") is not None:
                vmin = info["Valor mínimo"] / factor_escala
                indices_fuera_limite.extend(np.where(valores_1 < vmin)[0])
                ax_real.axhline(vmin, color='orange', linestyle='--', linewidth=1.2, alpha=0.7,
                                label=f'Mínimo ({info["Valor mínimo"]:.0f})')

            # if info.get("Valor máximo") is not None:
            #     vmax = info["Valor máximo"] / factor_escala
            #     indices_fuera_limite.extend(np.where(valores_1 > vmax)[0])
            #     ax_real.axhline(vmax, color='orange', linestyle='--', linewidth=1.2, alpha=0.7,
            #                     label=f'Máximo ({info["Valor máximo"]:.0f})')

            if info.get("Valor nominal") is not None:
                vnom = info["Valor nominal"] / factor_escala
                indices_fuera_limite.extend(np.where(valores_1 > vnom)[0])
                ax_real.axhline(vnom, color='yellow', linestyle='--', linewidth=1.2, alpha=0.7,
                                label=f'Nominal ({info["Valor nominal"]:.0f})')

            indices_fuera_limite = np.unique(indices_fuera_limite).astype(int)

            # Análisis por error de reconstrucción
            if var in errores_df.columns:
                errores_mae_var = errores_df[var].values

                if len(errores_mae_var) == len(valores_1):
                    indices_anom = np.where(errores_mae_var > umbral)[0]

                    if len(indices_anom) > 0:
                        indices_anom = np.array(indices_anom).astype(int)

                        # Puntos que están fuera de límite y con error alto → verde
                        indices_comb = np.intersect1d(indices_anom, indices_fuera_limite)
                        if len(indices_comb) > 0:
                            ax_real.scatter(indices_comb, valores_1[indices_comb],
                                            color="green", marker="o", s=50, label="Anomalía + Fuera de límite")

                        # Solo error alto → rojo
                        indices_solo_anom = np.setdiff1d(indices_anom, indices_comb)
                        if len(indices_solo_anom) > 0:
                            ax_real.scatter(indices_solo_anom, valores_1[indices_solo_anom],
                                            color="red", marker="x", s=50, label="Anomalía (Reconstrucción)")

                else:
                    st.write("⚠️ Error: longitud de errores_mae_var y valores_1 no coincide")

                
        # Mostrar leyenda
        ax_real.legend(loc='upper right')

        # === Gráfica 2: Error ===
        ax_err = axs[i, 1]
        ax_err.set_facecolor(fondo)
        ax_err.plot(errores_mae_var, color="red", linestyle="-", alpha=0.8, label="Error (MAE)")
        ax_err.set_title(f"Error de reconstrucción - {nombre}", fontweight="bold", color=color_letra)
        ax_err.axhline(y=umbral, color="red", linestyle="--", linewidth=1.5, label=f"Umbral de error : {umbral}")
        ax_err.tick_params(axis='both', labelsize=9, colors=color_letra)
        ax_err.grid(True, alpha=0.4)
        ax_err.legend(loc='upper right')


    plt.tight_layout()
    st.pyplot(fig)

    # Eliminar ejes vacíos
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

