import streamlit as st
import requests
import pandas as pd
from io import StringIO
import math
import time

BACKEND_URL = "http://localhost:8000/predict"
MAX_ROWS_PREVIEW = 100
BATCH_SIZE = 200  

st.set_page_config(page_title="Clasificador ODS", layout="centered")
st.title("Clasificador ODS – Front (texto o CSV)")

tab1, tab2 = st.tabs(["Texto manual", " Subir CSV"])

def call_predict(texts):
    payload = {"instancias": [{"texto": t} for t in texts]}
    r = requests.post(BACKEND_URL, json=payload, timeout=60)
    if r.status_code != 200:
        try:
            msg = r.json().get("error", f"HTTP {r.status_code}")
        except Exception:
            msg = f"HTTP {r.status_code}"
        raise RuntimeError(msg)
    return r.json()

def batch_predict(texts, batch_size=BATCH_SIZE):
    preds, probs = [], []
    total = len(texts)
    steps = math.ceil(total / batch_size)
    progress = st.progress(0.0, text="Procesando…")
    for i in range(steps):
        start = i * batch_size
        end = min((i+1) * batch_size, total)
        chunk = texts[start:end]
        data = call_predict(chunk)
        preds.extend(data.get("predicciones", []))
        p = data.get("probabilidades", [None] * len(chunk))
        probs.extend(p if p is not None else [None] * len(chunk))
        progress.progress((end/total), text=f"Procesado {end}/{total}")
        time.sleep(0.05)  
    progress.empty()
    return preds, probs

# Tab de texto
with tab1:
    st.write("Escribe uno o varios textos (uno por línea).")
    texts = st.text_area(
        "Textos",
        height=180,
        placeholder="Garantizar acceso a educación primaria...\nMejorar acceso a centros de salud...\nNecesitamos empleo digno...",
    )
    if st.button("Predecir (texto)"):
        lines = [t.strip() for t in texts.splitlines() if t.strip()]
        if not lines:
            st.error("Debes ingresar al menos un texto.")
        else:
            try:
                preds, probs = batch_predict(lines)
                df_out = pd.DataFrame({
                    "#": range(1, len(lines)+1),
                    "texto": lines,
                    "prediccion": preds,
                    "probabilidad": probs
                })
                st.success("Listo.")
                st.dataframe(df_out, use_container_width=True)
                csv = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Descargar resultados (CSV)", data=csv, file_name="predicciones_texto.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error llamando al backend: {e}")

# Tab para subir CSV
with tab2:
    st.write("Sube un CSV que contenga una columna con los textos.")
    file = st.file_uploader("Archivo CSV", type=["csv"])
    if file is not None:
 
        content = file.getvalue().decode("utf-8", errors="ignore")
        df = pd.read_csv(StringIO(content))
        st.write("Vista previa del archivo:")
        st.dataframe(df.head(MAX_ROWS_PREVIEW), use_container_width=True)

   
        cols = df.columns.tolist()
        col_text = st.selectbox("Selecciona la columna que contiene los textos", options=cols)

        if st.button("Predecir (CSV)"):
            texts_csv = df[col_text].astype(str).fillna("").tolist()
            texts_csv = [t.strip() for t in texts_csv if t.strip()]
            if not texts_csv:
                st.error("La columna seleccionada no contiene textos válidos.")
            else:
                try:
                    preds, probs = batch_predict(texts_csv)
                    
                    df_valid = pd.DataFrame({col_text: texts_csv})
                    df_valid["prediccion"] = preds
                    df_valid["probabilidad"] = probs
                    st.success(f"Listo. Filas procesadas: {len(df_valid)}")
                    st.dataframe(df_valid.head(MAX_ROWS_PREVIEW), use_container_width=True)

                    csv_out = df_valid.to_csv(index=False).encode("utf-8")
                    st.download_button(" Descargar resultados (CSV)", data=csv_out,
                                       file_name="predicciones_csv.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Error llamando al backend: {e}")

st.divider()
st.caption("Si el CSV es muy grande, se envía por lotes de 200 registros.")
