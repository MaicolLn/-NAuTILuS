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
from Secciones.nautilus import nautilus_en_marcha
from Secciones.prueba import nautilus_en_marcha_2
from Secciones.Panel import PanelC
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

if "health_index" not in st.session_state:
    st.session_state["health_index"] = {}

for nombre_subsistema in subsistemas:
    if nombre_subsistema not in st.session_state["health_index"]:
        st.session_state["health_index"][nombre_subsistema] = list(np.random.uniform(0.01, 3 * 0.1, 120))


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
    "🚀 Nautilus",
    "🚀 Nautilus en marcha",
    "Prueba"
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


elif seccion == "🚀 Nautilus en marcha":

    nautilus_en_marcha()


# -----------------------------------------------------
# NAUTILUS EN MARCHA
# -----------------------------------------------------

# elif seccion == "🚀 Nautilus en marcha":
#     from Secciones.nautilus import nautilus_en_marcha
#     nautilus_en_marcha(resultado, subsistemas)

elif seccion == "🚀 Nautilus":
    import numpy as np

    PanelC()

elif seccion == "Prueba":
    nautilus_en_marcha_2()
#     import numpy as np
#     # 1. Inicializa la lista si no existe
#     if "health_index" not in st.session_state:
#         st.session_state["health_index"] = list(np.random.uniform(0.01, 3 * 0.1, 100))  # valores iniciales

#     # 3. Llama a la función de graficado con todos los datos acumulados
#     PanelC()
