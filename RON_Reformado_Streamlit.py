#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ==========================================================
# APP STREAMLIT – RON REFORMADO (MODO PRO)
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

# UI limpia (sacar link y padding)
st.markdown("""
<style>
a[href^="#"] { display: none !important; }
.block-container { padding-top: 2rem; }
.big-font { font-size:22px !important; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ==========================================================
# HEADER
# ==========================================================

st.markdown("## 🧪 Estimación de RON - Reformado")

# ==========================================================
# CRITERIOS
# ==========================================================

REPRO_METODO = 0.83
UMBRAL_METODO = REPRO_METODO / 2
UMBRAL_METODO_SUP = 0.6

# ==========================================================
# MODELO
# ==========================================================

@st.cache_resource
def cargar_modelo():
    modelo = load("Modelo_RON_Reformado.joblib")
    columnas = load("Columnas_RON_Reformado.joblib")
    return modelo, columnas

try:
    RF, columnas_modelo = cargar_modelo()
    st.success("✅ Modelo Random Forest con validación metrológica")
except:
    st.error("❌ Error al cargar modelo")
    st.stop()

# ==========================================================
# INPUT
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
# BOTÓN
# ==========================================================

if archivo is not None:

    if st.button("🚀 Calcular RON"):

        with st.spinner("Procesando muestra..."):

            reformado = pd.read_csv(archivo, sep=";", encoding="latin1", header=None)

            try:
                celda_producto = reformado.loc[reformado[0] == "Producto", 4].values[0]
                celda_lims = reformado.loc[reformado[0] == "Número de Muestra", 4].values[0]
            except:
                st.error("❌ Formato de archivo inválido")
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

            if celda_producto == "REFORMADO_2201E":

                faltantes = [
                    k for k, v in datos_convertidos.items()
                    if isinstance(v, float) and np.isnan(v)
                ]

                if faltantes:
                    st.error("❌ Datos incompletos")
                    st.warning("Faltan ensayos:")
                    st.write(", ".join(faltantes))
                    st.stop()

            df_pred = pd.DataFrame([datos])[columnas_modelo]

            for col in df_pred.columns:
                df_pred[col] = (
                    df_pred[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .astype(float)
                )

            pred_arboles = np.array([
                tree.predict(df_pred)[0]
                for tree in RF.estimators_
            ])

            ron_estimado = np.round(pred_arboles.mean(), 1)
            ron_std = pred_arboles.std()

            # ======================================================
            # SEMÁFORO
            # ======================================================

            if ron_std <= UMBRAL_METODO:
                color = "green"
                estado = "ALTA CONFIABILIDAD"

            elif ron_std < UMBRAL_METODO_SUP:
                color = "orange"
                estado = "CONFIABILIDAD MEDIA"

            else:
                color = "red"
                estado = "BAJA CONFIABILIDAD"

            st.markdown(
                f"""
                <div style="
                    text-align: center;
                    font-size: 26px;
                    font-weight: bold;
                    color: {color};
                ">
                    ● {estado}
                </div>
                """,
                unsafe_allow_html=True
            )

            # ======================================================
            # RESULTADO VISUAL PRO
            # ======================================================

            if celda_producto == "REFORMADO_2201E":

                st.markdown("---")

                col1, col2 = st.columns(2)

                
                with col1:
                    st.markdown("### 🔢 RON estimado")

                    if ron_std < UMBRAL_METODO_SUP:
                        valor = str(ron_estimado).replace(".", ",")
                        color = "black"
                    else:
                        valor = "❌"
                        color = "red"

                    st.markdown(
                        f"""
                        <div style="
                            text-align: center;
                            font-size: 32px;
                            font-weight: bold;
                            color: {color};
                        ">
                            {valor}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                
                with col2:
                    st.markdown(f"### 📋 LIMS: {celda_lims}")

                st.markdown("---")

                # Semáforo grande
                st.markdown(
                    f"""
                    <div style="text-align:center;">
                        <h2 style="color:{color};">{icono} {estado}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.error("❌ Archivo no corresponde a REFORMADO")

