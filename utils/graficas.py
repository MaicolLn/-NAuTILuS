import math
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def graficar_modelos_comparados(
    resultado,
    df_1=None,
    df_2=None,
    seleccionadas=None,
    titulo="Generación sintética de datos con VAE",
    color_1="green",
    color_2="red",
    etiqueta_1="Datos sin anomalías",
    etiqueta_2="Datos con anomalías",
    color_real="purple",
    etiqueta_real="Datos reales",
    mostrar_reales=True,
    factor_escala=1,
    limites=None
):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import math
    import streamlit as st



    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("colorblind")

    # Validar variables
    variables = seleccionadas or list(set(df_1.columns if df_1 is not None else []).union(df_2.columns if df_2 is not None else []))
    variables = [v for v in variables if v in resultado]
    n_vars = len(variables)

    if n_vars == 0:
        st.warning("No hay variables válidas para graficar.")
        return

    n_cols = 2
    n_rows = math.ceil(n_vars / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows), sharex=False)
    axs = axs.flatten()

    for i, var in enumerate(variables):
        ax = axs[i]
        info = resultado.get(var, {})
        nombre = info.get("Nombre", var)

        # === Datos sintéticos
        valores_1 = df_1[var].sample(n=76).values / factor_escala if df_1 is not None and var in df_1.columns else None
        valores_2 = df_2[var].sample(n=76).values / factor_escala if df_2 is not None and var in df_2.columns else None

        # === Datos reales desde JSON
        valores_real = None
        if mostrar_reales:
            mediciones = info.get("Mediciones", None)
            if mediciones and isinstance(mediciones, list) and len(mediciones) >= 1:
                valores_real = mediciones[:76]
                valores_real = [v / factor_escala for v in valores_real]

        # === Plot
        if valores_real is not None:
            ax.plot(valores_real, color=color_real, linestyle="-", alpha=0.8, label=etiqueta_real)

        if valores_1 is not None:
            ax.plot(valores_1, color=color_1, linestyle="-", alpha=0.8, label=etiqueta_1)

        if valores_2 is not None:
            ax.plot(valores_2, color=color_2, linestyle="-", alpha=0.8, label=etiqueta_2)

        ax.set_title(nombre, fontsize=12, fontweight="bold")
        ax.set_ylabel(nombre, fontsize=10)
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(True, alpha=0.3)
    # Limitar el eje Y si hay límites definidos para la variable
        if limites and var in limites:
            ymin, ymax = limites[var]
            ax.set_ylim(ymin, ymax)

        # Líneas de referencia
        if info.get("Valor mínimo") is not None:
            vmin = info["Valor mínimo"] / factor_escala
            ax.axhline(vmin, color='orange', linestyle='--', linewidth=1.2, alpha=0.7, label=f'Mínimo ({info["Valor mínimo"]:.0f})')
        if info.get("Valor nominal") is not None:
            vnom = info["Valor nominal"] / factor_escala
            ax.axhline(vnom, color='yellow', linestyle='--', linewidth=1.2, alpha=0.7, label=f'Nominal ({info["Valor nominal"]:.0f})')
        if info.get("Valor máximo") is not None:
            vmax = info["Valor máximo"] / factor_escala
            ax.axhline(vmax, color='orange', linestyle='--', linewidth=1.2, alpha=0.7, label=f'Máximo ({info["Valor máximo"]:.0f})')

        ax.legend(fontsize=8, loc='upper right')

    # Eliminar ejes vacíos
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.suptitle(titulo, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig)
