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
# CONFIGURACIÓN
# ==========================================================

st.set_page_config(
    page_title="RON Reformado",
    page_icon="🧪",
    layout="centered"
)

# Ocultar icono 🔗
st.markdown("""
<style>
a[href^="#"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🧪 Estimación de RON - Reformado")
st.write("Modelo Random Forest con control metrológico")

# ==========================================================
# CRITERIOS METROLÓGICOS
# ==========================================================

REPRO_METODO = 0.83
UMBRAL_METODO = REPRO_METODO / 2
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
# SUBIDA ARCHIVO
# ==========================================================

archivo = st.file_uploader("📁 Cargar archivo CSV del LIMS", type=["csv"])

st.info("⬆️ Cargar archivo y luego presionar 'Calcular RON'")

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
# BOTÓN DE EJECUCIÓN
# ==========================================================

if archivo is not None:

    if st.button("🔍 Calcular RON"):

        try:
            reformado = pd.read_csv(archivo, sep=";", encoding="latin1", header=None)
        except:
            st.error("❌ Error al leer el archivo")
            st.stop()

        # IDENTIFICACIÓN
        try:
            celda_producto = reformado.loc[reformado[0] == "Producto", 4].values[0]
            celda_lims = reformado.loc[reformado[0] == "Número de Muestra", 4].values[0]
        except:
            st.error("❌ Formato de archivo no reconocido")
            st.stop()

        # VARIABLES
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

        # VALIDACIÓN
        if faltantes:
            st.error("❌ No se puede estimar el RON")
            st.warning("Faltan los siguientes ensayos:")
            st.write(", ".join(faltantes))
            st.stop()

        # PREPARAR DATAFRAME
        df_pred = pd.DataFrame([datos])[columnas_modelo]

        for col in df_pred.columns:
            df_pred[col] = (
                df_pred[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
                .astype(float)
            )

        # PREDICCIÓN
        pred_arboles = np.array([
            tree.predict(df_pred)[0]
            for tree in RF.estimators_
        ])

        ron_estimado = np.round(pred_arboles.mean(), 1)
        ron_std = pred_arboles.std()

        # CLASIFICACIÓN
        if ron_std <= UMBRAL_METODO:
            estado = "✅ ALTA confiabilidad"
        elif ron_std < UMBRAL_METODO_SUP:
            estado = "⚠️ Confiabilidad MEDIA"
        else:
            estado = "❌ BAJA confiabilidad - Verificar ASTM D2699"

        # RESULTADO
        st.subheader("📊 Resultado")

        if celda_producto == "REFORMADO_2201E":

            st.write(f"**LIMS:** {celda_lims}")

            if ron_std < UMBRAL_METODO_SUP:
                st.metric("RON estimado", str(ron_estimado).replace(".", ","))

            st.write(f"**Estado:** {estado}")

        else:
            st.error("❌ El archivo no corresponde a REFORMADO")

