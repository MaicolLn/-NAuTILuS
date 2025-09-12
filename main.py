import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sdv.single_table import TVAESynthesizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score

import json
import math

# -----------------------------------------------------
# Funciones
# -----------------------------------------------------
from utils.graficas import *
from Secciones.Gen_data import *
from Secciones.visualización import *
from Secciones.RUL import nautilus_en_marcha
from Secciones.deteccion_anom import detec_A
# -----------------------------------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------------------------------
st.set_page_config(page_title="NAUTILUS", layout="centered")

st.title("🔧 NAUTILUS")
st.markdown("### *Naval AI-based Utility for Targeted Intelligent Lifecycle Upkeep and Sustainability*")

st.image("motor.png", use_container_width=True, width=250, caption="Propulsor W26 Wärtsilä")


# Cargar subsistemas
with open("subsistemas.json", "r", encoding="utf-8") as f:
    subsistemas = json.load(f)
    st.session_state["subsistemas"] = subsistemas  # ✅ Esto evita el error

with open("umbrales_subsistemas.json", "r", encoding="utf-8") as f:
    umbrales_subsistemas= json.load(f)
    st.session_state["umbrales_subsistemas"] = umbrales_subsistemas # ✅ Esto evita el error

with open("modelosxsubsistemas.json", "r", encoding="utf-8") as f:
    modelosxsubsistemas= json.load(f)
    st.session_state["modelosxsubsistemas"] = modelosxsubsistemas # ✅ Esto evita el error
    
# Cargar umbrales
with open("umbrales.json", "r", encoding="utf-8") as f:
    umbrales = json.load(f)
    st.session_state["umbrales"] = umbrales  # ✅ Guarda los umbrales en el estado de sesión

# Cargar resultado.json (con límites y mediciones)
with open("resultado.json", "r", encoding="utf-8") as f:
    resultado = json.load(f)
if "resultado" not in st.session_state:
    try:
        with open("resultado.json", "r", encoding="utf-8") as f:
            st.session_state.resultado = json.load(f)
    except FileNotFoundError:
        st.warning("⚠️ No se encontró el archivo resultado.json")
        st.session_state.resultado = {}

import numpy as np
len_alet=31
if "health_index" not in st.session_state:
    st.session_state["health_index"] = {}

if "remaining_useful_life" not in st.session_state:
    st.session_state["remaining_useful_life"] = {}

for nombre_subsistema, variables in subsistemas.items():
    if nombre_subsistema not in st.session_state["remaining_useful_life"]:
        rul_dic = list(np.random.uniform(700, 1500, len_alet))
        st.session_state["remaining_useful_life"][nombre_subsistema] = []


for nombre_subsistema, variables in subsistemas.items():
    if nombre_subsistema not in st.session_state["health_index"]:
        max_valor = st.session_state["umbrales_subsistemas"].get(nombre_subsistema, 0.2)


        alet = list(np.random.uniform(max_valor*0.4, max_valor*0.8, len_alet))
        divisor = len(variables) if len(variables) > 0 else 1  # Evita división por cero
        alet_dividido = [x / divisor for x in alet]
        st.session_state["health_index"][nombre_subsistema] = alet

if "health_index_variables" not in st.session_state:
    st.session_state["health_index_variables"] = {}

for nombre_subsistema, variables in subsistemas.items():
    if nombre_subsistema not in st.session_state["health_index_variables"]:
        st.session_state["health_index_variables"][nombre_subsistema] = {}

    for var in variables:
        if var not in st.session_state["health_index_variables"][nombre_subsistema]:
            st.session_state["health_index_variables"][nombre_subsistema][var] =  alet_dividido

if "tema" not in st.session_state:
    st.session_state["tema"] = "Light"  # valor por defecto


# -----------------------------------------------------
# SESIÓN
# -----------------------------------------------------
if "modelo_VAE" not in st.session_state:
    st.session_state.modelo_VAE = None

if "datos_generados" not in st.session_state:
    st.session_state.datos_generados = None

# -----------------------------------------------------
# MENÚ PRINCIPAL (SELECCIÓN DE SECCIÓN)
# -----------------------------------------------------
st.divider()
seccion = st.sidebar.selectbox("📌 Selecciona una sección:", [
    "🔄 Generación de datos",
    "📉Detección de anomalías",
    "🚀Health index & RUL",
    # "🚀Health index & RUL_Prueba"
])


import json

# Cargar subsistemas desde archivo JSON
with open("subsistemas.json", "r", encoding="utf-8") as f:
    subsistemas = json.load(f)


# -----------------------------------------------------
# SIDEBAR DINÁMICO SEGÚN SECCIÓN
# -----------------------------------------------------
# st.sidebar.title("⚙️ Opciones")


# -----------------------------------------------------
# PANEL CENTRAL DE CONTENIDO SEGÚN SECCIÓN
# -----------------------------------------------------

if seccion == "🔄 Generación de datos":
    
    cargar_modelos()
# -----------------------------------------------------
# VISUALIZACIÓN
# -----------------------------------------------------
elif seccion == "📉Detección de anomalías":
    import numpy as np
    detec_A()


# elif seccion == "🚀Health index & RUL_p":

#     nautilus_en_marcha()


# -----------------------------------------------------
# NAUTILUS EN MARCHA
# -----------------------------------------------------

# elif seccion == "🚀 Nautilus en marcha":
#     from Secciones.nautilus import nautilus_en_marcha
#     nautilus_en_marcha(resultado, subsistemas)

# elif seccion == "🚀 Nautilus univariable":
#     import numpy as np

#     PanelC()

elif seccion == "🚀Health index & RUL":
    nautilus_en_marcha()
#     import numpy as np
#     # 1. Inicializa la lista si no existe
#     if "health_index" not in st.session_state:
#         st.session_state["health_index"] = list(np.random.uniform(0.01, 3 * 0.1, 100))  # valores iniciales

#     # 3. Llama a la función de graficado con todos los datos acumulados
#     PanelC()
