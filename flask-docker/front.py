import streamlit as st
import requests
import pandas as pd
import numpy as np
from io import StringIO
import json
import math
import time
import matplotlib.pyplot as plt
import seaborn as sns  # pip install seaborn

# ------------------------- CONFIG -------------------------
BACKEND_URL_PREDICT     = "http://localhost:8000/predict"
BACKEND_URL_RETRAIN     = "http://localhost:8000/retrain"
BACKEND_URL_METRICS     = "http://localhost:8000/metrics"
BACKEND_URL_HISTORICAL  = "http://localhost:8000/historical_data"

MAX_ROWS_PREVIEW = 100
BATCH_SIZE = 200

st.set_page_config(page_title="Clasificador ODS", layout="centered")
st.title("Clasificador ODS – Front")

# ------------------------- HELPERS HTTP -------------------------
def post_json(url, payload, timeout=120):
    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code != 200:
        try:
            msg = r.json().get("error", f"HTTP {r.status_code}")
        except Exception:
            msg = f"HTTP {r.status_code}"
        raise RuntimeError(msg)
    return r.json()

def get_json(url, timeout=60):
    r = requests.get(url, timeout=timeout)
    if r.status_code != 200:
        try:
            msg = r.json().get("error", f"HTTP {r.status_code}")
        except Exception:
            msg = f"HTTP {r.status_code}"
        raise RuntimeError(msg)
    return r.json()

# ------------------------- HELPERS PREDICT -------------------------
def call_predict(texts):
    payload = {"instancias": [{"textos": t} for t in texts]}
    return post_json(BACKEND_URL_PREDICT, payload, timeout=120)

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
        time.sleep(0.02)
    progress.empty()
    return preds, probs

# ------------------------- HELPERS RETRAIN -------------------------
def call_retrain(instancias):
    """
    instancias: lista de dicts {'textos': <str>, 'labels': <int|str>}
    """
    payload = {"instancias": instancias}
    return post_json(BACKEND_URL_RETRAIN, payload, timeout=240)

# ------------------------- PARSEADORES DE ENTRADA -------------------------
def parse_csv_or_excel(file):
    """
    Detecta por extensión. Devuelve DataFrame.
    """
    name = file.name.lower()
    content = file.getvalue()
    if name.endswith(".csv"):
        df = pd.read_csv(StringIO(content.decode("utf-8", errors="ignore")))
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(content)  # requiere openpyxl para .xlsx
    else:
        raise ValueError("Formato no soportado. Sube CSV, XLSX o XLS.")
    return df

def parse_json_predict(file_or_str):
    """
    Acepta:
      - archivo JSON o string con:
        * {"instancias": [{"textos": "..."} , ...]}
        * [{"textos": "..."}, ...]
        * {"textos": ["..",".."]}
    Devuelve lista de textos.
    """
    if hasattr(file_or_str, "getvalue"):
        raw = file_or_str.getvalue().decode("utf-8", errors="ignore")
    else:
        raw = str(file_or_str)
    data = json.loads(raw)

    if isinstance(data, dict) and "instancias" in data:
        textos = [str(it["textos"]).strip() for it in data["instancias"] if "textos" in it and str(it["textos"]).strip() != ""]
    elif isinstance(data, list):
        textos = [str(it.get("textos", "")).strip() for it in data if isinstance(it, dict) and str(it.get("textos","")).strip() != ""]
    elif isinstance(data, dict) and "textos" in data and isinstance(data["textos"], list):
        textos = [str(t).strip() for t in data["textos"] if str(t).strip() != ""]
    else:
        raise ValueError("JSON de predicción no reconocido.")
    if not textos:
        raise ValueError("JSON sin textos válidos.")
    return textos

def parse_json_retrain(file_or_str):
    """
    Acepta:
      - archivo JSON o string con:
        * {"instancias": [{"textos": "...", "labels": ...}, ...]}
        * [{"textos": "...", "labels": ...}, ...]
    Devuelve lista de dicts {'textos', 'labels'}.
    """
    if hasattr(file_or_str, "getvalue"):
        raw = file_or_str.getvalue().decode("utf-8", errors="ignore")
    else:
        raw = str(file_or_str)
    data = json.loads(raw)

    if isinstance(data, dict) and "instancias" in data:
        items = data["instancias"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("JSON de reentrenamiento no reconocido.")

    instancias = []
    for it in items:
        if not isinstance(it, dict): 
            continue
        t = str(it.get("textos", "")).strip()
        l = it.get("labels", None)
        if t != "" and l is not None and str(l).strip() != "":
            try:
                l = int(l) if str(l).isdigit() else l
            except:
                pass
            instancias.append({"textos": t, "labels": l})

    if not instancias:
        raise ValueError("JSON sin instancias válidas (textos+labels).")
    return instancias

# ------------------------- GRAFICADO -------------------------
def labels_from_report(report_dict):
    """
    Toma las clases desde classification_report (excluye accuracy y promedios).
    """
    keys = [k for k in report_dict.keys() if "avg" not in k and k != "accuracy"]
    try:
        keys_sorted = sorted(keys, key=lambda x: int(x))  # intenta ordenar numéricamente
    except:
        keys_sorted = sorted(keys)
    return keys_sorted

def plot_confusion_matrix(cm, class_labels, title="Matriz de confusión"):
    cm = np.array(cm)
    if len(class_labels) != cm.shape[0]:
        class_labels = [str(i) for i in range(cm.shape[0])]
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", cbar=False,
                xticklabels=class_labels, yticklabels=class_labels, ax=ax)
    ax.set_xlabel("Etiqueta predicha")
    ax.set_ylabel("Etiqueta real")
    ax.set_title(title)
    plt.tight_layout()
    st.pyplot(fig)

def show_classification_report_table(report_dict, show_accuracy_as_caption=True):
    rep = {k: v for k, v in report_dict.items()}
    acc = rep.pop("accuracy", None)
    df_rep = pd.DataFrame(rep).T
    # redondear columnas numéricas
    for col in df_rep.columns:
        try:
            df_rep[col] = pd.to_numeric(df_rep[col])
            df_rep[col] = df_rep[col].map(lambda x: round(float(x), 4))
        except:
            pass
    st.write("### Reporte de clasificación")
    st.dataframe(df_rep, use_container_width=True)
    # Accuracy menos llamativo
    if acc is not None:
        try:
            acc_val = float(acc)
            if show_accuracy_as_caption:
                st.caption(f"Accuracy global: {acc_val:.4f}")
            else:
                st.info(f"Accuracy global: {acc_val:.4f}")
        except:
            pass

def plot_class_balance_from_report(report_dict, title="Balance de clases (support)"):
    """
    Construye un gráfico de barras usando el 'support' del classification_report.
    """
    # Extrae clases y supports (omitir accuracy y promedios)
    items = []
    for k, v in report_dict.items():
        if k == "accuracy" or "avg" in k:
            continue
        if isinstance(v, dict) and "support" in v:
            items.append((str(k), int(v["support"])))
    if not items:
        st.info("No se encontró información de 'support' en el reporte.")
        return
    labels, supports = zip(*items)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=list(labels), y=list(supports), ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Clase")
    ax.set_ylabel("Soporte (n)")
    plt.tight_layout()
    st.pyplot(fig)

# ------------------------- LAYOUT: 4 TABS -------------------------
tab_pred, tab_retrain, tab_metrics, tab_hist = st.tabs(["Predecir", "Reentrenar", "Métricas", "Histórico"])

# ------------------------- TAB: PREDECIR -------------------------
with tab_pred:
    modo = st.radio(
        "Elige cómo enviar datos",
        ["Editor manual", "Subir CSV/Excel", "Subir JSON"],
        horizontal=True,
        key="pred_radio_mode"
    )

    if modo == "Editor manual":
        st.info("Edita la tabla. Debe tener la columna: textos.")
        df_edit = st.data_editor(
            pd.DataFrame({"textos": [""]}),
            num_rows="dynamic",
            use_container_width=True,
            key="pred_editor_table",
        )
        if st.button("Predecir", key="pred_editor_btn"):
            df_clean = df_edit.copy()
            df_clean["textos"] = df_clean["textos"].astype(str).str.strip()
            df_clean = df_clean[(df_clean["textos"] != "")]
            if df_clean.empty:
                st.error("Agrega al menos una fila con 'textos'.")
            else:
                try:
                    textos = df_clean["textos"].tolist()
                    preds, probs = batch_predict(textos)
                    df_out = pd.DataFrame({
                        "#": range(1, len(textos)+1),
                        "texto": textos,
                        "prediccion": preds,
                        "probabilidad": probs
                    })
                    st.success("Listo.")
                    st.dataframe(df_out, use_container_width=True)
                    csv = df_out.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Descargar resultados (CSV)", data=csv,
                                       file_name="predicciones_texto.csv", mime="text/csv",
                                       key="pred_editor_download")
                except Exception as e:
                    st.error(f"Error llamando al backend: {e}")

    elif modo == "Subir CSV/Excel":
        file = st.file_uploader("Archivo CSV/XLSX/XLS", type=["csv","xlsx","xls"], key="pred_file")
        if file is not None:
            try:
                df = parse_csv_or_excel(file)
                st.write("Vista previa del archivo:")
                st.dataframe(df.head(MAX_ROWS_PREVIEW), use_container_width=True)
                cols = df.columns.tolist()
                col_text = st.selectbox("Columna de textos", options=cols, key="pred_cols")
                if st.button("Predecir (archivo)", key="pred_file_btn"):
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
                                               file_name="predicciones_csv.csv", mime="text/csv",
                                               key="pred_file_download")
                        except Exception as e:
                            st.error(f"Error llamando al backend: {e}")
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")

    else:  # Subir JSON
        filej = st.file_uploader("Archivo JSON", type=["json"], key="pred_json")
        pasted = st.text_area("…o pega JSON aquí", height=160, key="pred_json_text")
        if st.button("Predecir (JSON)", key="pred_json_btn"):
            try:
                if filej is not None:
                    textos = parse_json_predict(filej)
                else:
                    textos = parse_json_predict(pasted)
                preds, probs = batch_predict(textos)
                df_out = pd.DataFrame({"texto": textos, "prediccion": preds, "probabilidad": probs})
                st.success(f"Listo. Filas procesadas: {len(df_out)}")
                st.dataframe(df_out.head(MAX_ROWS_PREVIEW), use_container_width=True)
                csv = df_out.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Descargar resultados (CSV)", data=csv,
                                   file_name="predicciones_json.csv", mime="text/csv",
                                   key="pred_json_download")
            except Exception as e:
                st.error(f"Error en JSON: {e}")

# ------------------------- TAB: REENTRENAR -------------------------
with tab_retrain:
    modo_rt = st.radio(
        "Elige cómo enviar datos",
        ["Editor manual", "Subir CSV/Excel", "Subir JSON"],
        horizontal=True,
        key="retrain_radio_mode"
    )

    if modo_rt == "Editor manual":
        st.info("Edita la tabla. Debe tener columnas: textos, labels.")
        df_edit = st.data_editor(
            pd.DataFrame({"textos": [""], "labels": [""]}),
            num_rows="dynamic",
            use_container_width=True,
            key="rt_editor",
        )
        if st.button("Reentrenar", key="rt_editor_btn"):
            df_clean = df_edit.copy()
            df_clean["textos"] = df_clean["textos"].astype(str).str.strip()
            df_clean["labels"] = df_clean["labels"].astype(str).str.strip()
            df_clean = df_clean[(df_clean["textos"] != "") & (df_clean["labels"] != "")]
            if df_clean.empty:
                st.error("Agrega al menos una fila con 'textos' y 'labels'.")
            else:
                try:
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
                        report = metricas.get("reporte_clasificacion") or resp.get("reporte_clasificacion")
                        if cm is not None and report is not None:
                            labels = labels_from_report(report)
                            st.write("### Matriz de confusión")
                            plot_confusion_matrix(cm, labels, title="Matriz de confusión (post-retrain)")
                            show_classification_report_table(report)
                            # Balance SOLO aquí
                            st.write("### Balance de clases")
                            plot_class_balance_from_report(report, title="Balance de clases (support) – post-retrain")

                    # Descarga de instancias usadas
                    jsonl = "\n".join(pd.DataFrame(instancias).apply(lambda r: r.to_json(force_ascii=False), axis=1))
                    st.download_button("⬇️ Descargar instancias (JSONL)",
                                       data=jsonl.encode("utf-8"),
                                       file_name="retrain_instancias.jsonl",
                                       mime="application/json",
                                       key="rt_editor_download")
                except Exception as e:
                    st.error(f"Error reentrenando: {e}")

    elif modo_rt == "Subir CSV/Excel":
        file_rt = st.file_uploader("Archivo CSV/XLSX/XLS con columnas textos y labels", type=["csv", "xlsx", "xls"], key="rt_file")
        if file_rt is not None:
            try:
                df_rt = parse_csv_or_excel(file_rt)
                st.write("Vista previa:")
                st.dataframe(df_rt.head(MAX_ROWS_PREVIEW), use_container_width=True)

                cols_rt = df_rt.columns.tolist()
                idx_text = cols_rt.index("textos") if "textos" in cols_rt else 0
                idx_lab  = cols_rt.index("labels") if "labels" in cols_rt else (1 if len(cols_rt) > 1 else 0)

                col_text_rt  = st.selectbox("Columna de textos", options=cols_rt, index=idx_text, key="rt_cols_text")
                col_label_rt = st.selectbox("Columna de labels", options=cols_rt, index=idx_lab, key="rt_cols_label")

                if st.button("Reentrenar (archivo)", key="rt_file_btn"):
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
                                report = metricas.get("reporte_clasificacion") or resp.get("reporte_clasificacion")
                                if cm is not None and report is not None:
                                    labels = labels_from_report(report)
                                    st.write("### Matriz de confusión")
                                    plot_confusion_matrix(cm, labels, title="Matriz de confusión (post-retrain)")
                                    show_classification_report_table(report)
                                    # Balance SOLO aquí
                                    st.write("### Balance de clases")
                                    plot_class_balance_from_report(report, title="Balance de clases (support) – post-retrain")

                            # Descarga de instancias usadas
                            df_used = pd.DataFrame(instancias)
                            jsonl = "\n".join(df_used.apply(lambda r: r.to_json(force_ascii=False), axis=1))
                            st.download_button("⬇️ Descargar instancias (JSONL)",
                                               data=jsonl.encode("utf-8"),
                                               file_name="retrain_instancias.jsonl",
                                               mime="application/json",
                                               key="rt_file_download")
                        except Exception as e:
                            st.error(f"Error reentrenando: {e}")
            except Exception as e:
                st.error(f"No se pudo leer el archivo: {e}")

    else:  # Subir JSON
        filej_rt = st.file_uploader("Archivo JSON", type=["json"], key="rt_json")
        pasted_rt = st.text_area("…o pega JSON aquí", height=160, key="rt_json_text")
        if st.button("Reentrenar (JSON)", key="rt_json_btn"):
            try:
                if filej_rt is not None:
                    instancias = parse_json_retrain(filej_rt)
                else:
                    instancias = parse_json_retrain(pasted_rt)
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
                    report = metricas.get("reporte_clasificacion") or resp.get("reporte_clasificacion")
                    if cm is not None and report is not None:
                        labels = labels_from_report(report)
                        st.write("### Matriz de confusión")
                        plot_confusion_matrix(cm, labels, title="Matriz de confusión (post-retrain)")
                        show_classification_report_table(report)
                        # Balance SOLO aquí
                        st.write("### Balance de clases")
                        plot_class_balance_from_report(report, title="Balance de clases (support) – post-retrain")
            except Exception as e:
                st.error(f"Error en JSON: {e}")

# ------------------------- TAB: MÉTRICAS -------------------------
with tab_metrics:
    st.write(
        "Métricas del **modelo actual** tras el **último reentrenamiento** "
        "(calculadas sobre el set de validación configurado en el backend)."
    )
    if st.button("Calcular métricas del modelo actual", key="metrics_btn"):
        try:
            resp = get_json(BACKEND_URL_METRICS)
            metricas = resp.get("metricas", {})
            if not metricas:
                st.info("El backend no devolvió métricas.")
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Precision (macro)", metricas.get("precision"))
                col2.metric("Recall (macro)",    metricas.get("recall"))
                col3.metric("F1 (macro)",        metricas.get("f1"))

                cm = metricas.get("matriz_confusion")
                report = metricas.get("reporte_clasificacion")
                if cm is not None and report is not None:
                    labels = labels_from_report(report)
                    st.write("### Matriz de confusión")
                    plot_confusion_matrix(cm, labels, title="Matriz de confusión (modelo actual)")
                    show_classification_report_table(report)
                    # Balance SOLO aquí
                    st.write("### Balance de clases")
                    plot_class_balance_from_report(report, title="Balance de clases (support) – modelo actual")
        except Exception as e:
            st.error(f"Error consultando métricas: {e}")

# ------------------------- TAB: HISTÓRICO -------------------------
with tab_hist:
    st.write("Estos son los **datos con los que se reentrenó el modelo**. Verifica si se agregaron los de la última ejecución.")
    if st.button("Cargar histórico", key="hist_simple_btn"):
        try:
            payload = get_json(BACKEND_URL_HISTORICAL)
            filas = payload.get("instancias", [])
            if not filas:
                st.info("No hay datos en el histórico.")
            else:
                df_hist = pd.DataFrame(filas)
                st.dataframe(df_hist, use_container_width=True)
                st.caption(f"Total filas en histórico: {len(df_hist)}")
        except Exception as e:
            st.error(f"Error consultando histórico: {e}")