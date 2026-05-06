#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ==========================================================
# APP STREAMLIT – ESTIMACIÓN DE RON REFORMADO
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import warnings

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================================

st.set_page_config(
    page_title="RON Reformado",
    page_icon="🧪",
    layout="centered"
)

st.title("🧪 Estimación de RON - Reformado")
st.write("Modelo Random Forest con control metrológico")

# ==========================================================
# CRITERIOS METROLÓGICOS
# ==========================================================

REPRO_METODO = 0.83
UMBRAL_METODO = REPRO_METODO / 2   # ≈ 0.42
UMBRAL_METODO_SUP = 0.6

# ==========================================================
# CARGA DEL MODELO
# ==========================================================

@st.cache_resource
def cargar_modelo():
    modelo = load("Modelo_RON_Reformado.joblib")
    columnas = load("Columnas_RON_Reformado.joblib")
    return modelo, columnas

try:
    RF, columnas_modelo = cargar_modelo()
    st.success("✅ Modelo cargado correctamente")
except:
    st.error("❌ No se pudo cargar el modelo")
    st.stop()

# ==========================================================
# SUBIDA DE ARCHIVO
# ==========================================================

archivo = st.file_uploader("📁 Cargar archivo CSV del LIMS", type=["csv"])

# ==========================================================
# FUNCIONES
# ==========================================================

def extraer_valor(df, nombre):
    fila = df[df[1] == nombre]
    if fila.empty:
        return np.nan
    return fila.iloc[0, 4]

def convertir(valor):
    if isinstance(valor, str):
        try:
            return float(valor.replace(",", "."))
        except:
            return np.nan
    return valor

# ==========================================================
# PROCESAMIENTO
# ==========================================================

if archivo is not None:

    try:
        reformado = pd.read_csv(archivo, sep=";", encoding="latin1", header=None)
    except:
        st.error("❌ Error al leer el archivo. Verifique formato CSV.")
        st.stop()

    # ======================================================
    # IDENTIFICACIÓN
    # ======================================================

    try:
        celda_producto = reformado.loc[reformado[0] == "Producto", 4].values[0]
        celda_lims = reformado.loc[reformado[0] == "Número de Muestra", 4].values[0]
    except:
        st.error("❌ No se reconoce el formato del archivo")
        st.stop()

    # ======================================================
    # EXTRACCIÓN DE VARIABLES
    # ======================================================

    datos = {
        'DENSIDAD': extraer_valor(reformado, "Densidad a 15ºC"),
        'IBP': extraer_valor(reformado, "IBP"),
        'T5': extraer_valor(reformado, "5% vol"),
        'T10': extraer_valor(reformado, "10% vol"),
        'T20': extraer_valor(reformado, "20% vol"),
        'T30': extraer_valor(reformado, "30% vol"),
        'T40': extraer_valor(reformado, "40% vol"),
        'T50': extraer_valor(reformado, "50% vol"),
        'T60': extraer_valor(reformado, "60% vol"),
        'T70': extraer_valor(reformado, "70% vol"),
        'T80': extraer_valor(reformado, "80% vol"),
        'T90': extraer_valor(reformado, "90% vol"),
        'T95': extraer_valor(reformado, "95% vol"),
        'PUNTO FINAL': extraer_valor(reformado, "Punto Final"),
        'TENS VAP': extraer_valor(reformado, "Tensión de Vapor")
    }

    datos_convertidos = {k: convertir(v) for k, v in datos.items()}

    faltantes = [
        k for k, v in datos_convertidos.items()
        if isinstance(v, float) and np.isnan(v)
    ]

    # ======================================================
    # VALIDACIÓN DE DATOS
    # ======================================================

    if faltantes:
        st.error("❌ No se puede estimar el RON")
        st.warning("Faltan los siguientes ensayos:")
        st.write(", ".join(faltantes))
        st.stop()

    # ======================================================
    # PREPARAR DATAFRAME
    # ======================================================

    df_pred = pd.DataFrame([datos])[columnas_modelo]

    for col in df_pred.columns:
        df_pred[col] = (
            df_pred[col]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

    # ======================================================
    # PREDICCIÓN RANDOM FOREST
    # ======================================================

    pred_arboles = np.array([
        tree.predict(df_pred)[0]
        for tree in RF.estimators_
    ])

    ron_estimado = np.round(pred_arboles.mean(), 1)
    ron_std = pred_arboles.std()

    # ======================================================
    # CLASIFICACIÓN METROLÓGICA
    # ======================================================

    if ron_std <= UMBRAL_METODO:
        estado = "✅ ALTA confiabilidad"
    elif ron_std < UMBRAL_METODO_SUP:
        estado = "⚠️ Confiabilidad MEDIA (interpretar con precaución)"
    else:
        estado = "❌ BAJA confiabilidad (recomendada ASTM D2699)"

    # ======================================================
    # RESULTADO FINAL
    # ======================================================

    st.subheader("📊 Resultado")

    if celda_producto == "REFORMADO_2201E":

        st.write(f"**LIMS:** {celda_lims}")

        if ron_std < UMBRAL_METODO_SUP:
            st.metric("RON estimado", str(ron_estimado).replace(".", ","))

        st.write(f"**Estado:** {estado}")
        st.write(f"**Desvío (std RF):** {round(ron_std,3)}")

    else:
        st.error("❌ El archivo no corresponde a REFORMADO")

