import streamlit as st
import requests
import pandas as pd
from io import StringIO
import math
import time

BACKEND_URL = "http://localhost:8000/predict"
BACKEND_URL_RETRAIN = "http://localhost:8000/retrain"
MAX_ROWS_PREVIEW = 100
BATCH_SIZE = 200

st.set_page_config(page_title="Clasificador ODS", layout="centered")
st.title("Clasificador ODS – Front (texto o CSV)")

# Ahora 3 tabs
tab1, tab2, tab3 = st.tabs(["Texto manual", " Subir CSV", "Reentrenar (JSON/CSV)"])

# ------------------------- utilidades predict -------------------------
def call_predict(texts):
    payload = {"instancias": [{"textos": t} for t in texts]}
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

# ------------------------- TAB 1: texto manual -------------------------
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

# ------------------------- TAB 2: subir CSV para predecir -------------------------
with tab2:
    st.write("Sube un CSV que contenga una columna con los textos.")
    file = st.file_uploader("Archivo CSV", type=["csv"], key="predict_csv")
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
                    st.download_button("⬇️ Descargar resultados (CSV)", data=csv_out,
                                       file_name="predicciones_csv.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"Error llamando al backend: {e}")

# ------------------------- utilidades retrain -------------------------
def call_retrain(instancias):
    """
    instancias: lista de dicts con llaves {'textos': <str>, 'labels': <int|str>}
    """
    payload = {"instancias": instancias}
    r = requests.post(BACKEND_URL_RETRAIN, json=payload, timeout=120)
    if r.status_code != 200:
        try:
            msg = r.json().get("error", f"HTTP {r.status_code}")
        except Exception:
            msg = f"HTTP {r.status_code}"
        raise RuntimeError(msg)
    return r.json()

# ------------------------- TAB 3: reentrenar -------------------------
with tab3:
    st.write("Reentrena el modelo enviando instancias con **textos** y **labels**.")
    modo = st.radio("Elige cómo enviar datos", ["Editor manual", "Subir CSV"], horizontal=True)

    if modo == "Editor manual":
        st.info("Edita la tabla y pulsa **Reentrenar**. Debe tener columnas: textos, labels.")
        df_edit = st.data_editor(
            pd.DataFrame({"textos": [""], "labels": [""]}),
            num_rows="dynamic",
            use_container_width=True,
            key="retrain_editor",
        )

        if st.button("Reentrenar (editor)"):
            df_clean = df_edit.copy()
            df_clean["textos"] = df_clean["textos"].astype(str).str.strip()
            df_clean["labels"] = df_clean["labels"].astype(str).str.strip()
            df_clean = df_clean[(df_clean["textos"] != "") & (df_clean["labels"] != "")]
            if df_clean.empty:
                st.error("Agrega al menos una fila con 'textos' y 'labels'.")
            else:
                try:
                    # convierte labels a int si son dígitos, si no deja string
                    instancias = [
                        {"textos": t, "labels": (int(l) if str(l).isdigit() else l)}
                        for t, l in zip(df_clean["textos"].tolist(), df_clean["labels"].tolist())
                    ]
                    with st.spinner("Reentrenando…"):
                        resp = call_retrain(instancias)
                    st.success(resp.get("mensaje", "Modelo reentrenado"))
                    metricas = resp.get("metricas")
                    if metricas:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Precision (macro)", metricas.get("precision"))
                        col2.metric("Recall (macro)",    metricas.get("recall"))
                        col3.metric("F1 (macro)",        metricas.get("f1"))
                        cm = metricas.get("matriz_confusion")
                        if cm is not None:
                            st.write("### Matriz de confusión")
                            st.dataframe(pd.DataFrame(cm), use_container_width=True)

                    # descarga de las instancias usadas
                    jsonl = "\n".join(pd.DataFrame(instancias).apply(lambda r: r.to_json(force_ascii=False), axis=1))
                    st.download_button("⬇️ Descargar instancias usadas (JSONL)",
                                       data=jsonl.encode("utf-8"),
                                       file_name="retrain_instancias.jsonl",
                                       mime="application/json")
                except Exception as e:
                    st.error(f"Error reentrenando: {e}")

    else:
        st.write("Sube un CSV con columnas **textos** y **labels** (o mapea columnas).")
        file_rt = st.file_uploader("Archivo CSV (reentrenamiento)", type=["csv"], key="csv_retrain")
        if file_rt is not None:
            content_rt = file_rt.getvalue().decode("utf-8", errors="ignore")
            try:
                df_rt = pd.read_csv(StringIO(content_rt))
            except Exception as e:
                st.error(f"No se pudo leer el CSV: {e}")
                df_rt = None

            if df_rt is not None:
                st.write("Vista previa:")
                st.dataframe(df_rt.head(MAX_ROWS_PREVIEW), use_container_width=True)

                cols_rt = df_rt.columns.tolist()
                # preseleccionar si ya existen "textos" y "labels"
                idx_text = cols_rt.index("textos") if "textos" in cols_rt else 0
                idx_lab  = cols_rt.index("labels") if "labels" in cols_rt else (1 if len(cols_rt) > 1 else 0)

                col_text_rt  = st.selectbox("Columna de textos", options=cols_rt, index=idx_text)
                col_label_rt = st.selectbox("Columna de labels", options=cols_rt, index=idx_lab)

                if st.button("Reentrenar (CSV)"):
                    sub = df_rt[[col_text_rt, col_label_rt]].copy()
                    sub[col_text_rt]  = sub[col_text_rt].astype(str).str.strip()
                    sub[col_label_rt] = sub[col_label_rt].astype(str).str.strip()
                    sub = sub[(sub[col_text_rt] != "") & (sub[col_label_rt] != "")]
                    if sub.empty:
                        st.error("No hay filas válidas tras limpieza.")
                    else:
                        try:
                            instancias = [
                                {"textos": t, "labels": (int(l) if str(l).isdigit() else l)}
                                for t, l in zip(sub[col_text_rt].tolist(), sub[col_label_rt].tolist())
                            ]
                            with st.spinner("Reentrenando…"):
                                resp = call_retrain(instancias)
                            st.success(resp.get("mensaje", "Modelo reentrenado"))
                            metricas = resp.get("metricas")
                            if metricas:
                                col1, col2, col3 = st.columns(3)
                                col1.metric("Precision (macro)", metricas.get("precision"))
                                col2.metric("Recall (macro)",    metricas.get("recall"))
                                col3.metric("F1 (macro)",        metricas.get("f1"))
                                cm = metricas.get("matriz_confusion")
                                if cm is not None:
                                    st.write("### Matriz de confusión")
                                    st.dataframe(pd.DataFrame(cm), use_container_width=True)

                            # descarga de las instancias usadas
                            df_used = pd.DataFrame(instancias)
                            jsonl = "\n".join(df_used.apply(lambda r: r.to_json(force_ascii=False), axis=1))
                            st.download_button("⬇️ Descargar instancias usadas (JSONL)",
                                               data=jsonl.encode("utf-8"),
                                               file_name="retrain_instancias.jsonl",
                                               mime="application/json")
                        except Exception as e:
                            st.error(f"Error reentrenando: {e}")

st.divider()
st.caption("Si el CSV es muy grande, se envía por lotes de 200 registros para predicción. El reentrenamiento envía todo en una sola solicitud JSON.")