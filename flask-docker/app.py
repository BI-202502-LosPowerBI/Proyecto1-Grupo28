from flask import Flask, request, jsonify
import pandas as pd
import os, joblib, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "modelo_ods.joblib")
DATA_DIR = os.path.join(BASE_DIR, "data")
VALIDATION_DATA_PATH = os.path.join(DATA_DIR, "datos_validacion.xlsx")
HISTORICAL_DATA_PATH = os.path.join(DATA_DIR, "datos_historicos.xlsx")

# Se carga el Pipeline entrenado
try:
    model = joblib.load(MODEL_PATH)
    print(f"Modelo cargado: {MODEL_PATH}")

except Exception as e:
    print(f"Error cargando modelo en {MODEL_PATH}: {e}")
    model = None

# Endpoint para verificar que el servicio está activo
@app.get("/ping")
def ping():
    return jsonify(ok=True)

# Endpoint para predecir nuevas instancias
@app.post("/predict")
def predict():
    if model is None:
        return jsonify(error="Modelo no cargado"), 500
    
    data = request.get_json(force=True)

    if "instancias" not in data:
        return jsonify(error="Formato inválido. Se requiere 'instancias'"), 400
    
    for it in data["instancias"]:
        if "textos" not in it:
            return jsonify(error="Formato inválido. Cada instancia requiere 'textos'"), 400
        
    textos = [it["textos"] for it in data["instancias"]]
    preds = model.predict(textos).tolist()
    probs = None

    try:
        proba = model.predict_proba(textos)
        idx = np.argmax(proba, axis=1)
        probs = [float(proba[i, idx[i]]) for i in range(len(textos))]

    except Exception:
        return jsonify(error="Error prediciendo probabilidades"), 500

    return jsonify(predicciones=preds, probabilidades=probs)

# Endpoint para reentrenar el modelo con nuevas instancias
@app.post("/retrain")
def retrain():
    global model
    if model is None:
        return jsonify(error="Modelo no cargado"), 500
    
    data = request.get_json(force=True)

    if "instancias" not in data:
        return jsonify(error="Formato inválido. Se requiere 'instancias'"), 400
    
    for it in data["instancias"]:
        if "textos" not in it or "labels" not in it:
            return jsonify(error="Formato inválido. Cada instancia requiere 'textos' y 'labels'"), 400

    df_historico = pd.read_excel(HISTORICAL_DATA_PATH, engine="openpyxl")
    df_nuevo = pd.DataFrame(data["instancias"])
    df = pd.concat([df_historico, df_nuevo], ignore_index=True)

    try:
        tfidf: TfidfVectorizer = model.named_steps["tfidf"]
        clf: LogisticRegression = model.named_steps["clf"]

        try:
            tfidf.fit(df["textos"])
            textos_tfidf = tfidf.transform(df["textos"])
            clf.fit(textos_tfidf, df["labels"])
        except Exception as e:
            return jsonify(error=f"Error reentrenando cada modelo: {e}"), 500

        joblib.dump(model, MODEL_PATH)
        df.to_excel(HISTORICAL_DATA_PATH, index=False, engine="openpyxl")

        df = pd.read_excel(VALIDATION_DATA_PATH, engine="openpyxl")
        
        X = df["textos"].tolist()
        y = df["labels"].tolist()

        y_pred = model.predict(X)

        precision_macro, recall_macro, f1_macro = (
            precision_score(y, y_pred, average="macro", zero_division=0),
            recall_score(y, y_pred, average="macro", zero_division=0),
            f1_score(y, y_pred, average="macro", zero_division=0),
        )

        cm = confusion_matrix(y, y_pred).tolist()

    except Exception as e:
        return jsonify(error=f"Error reentrenando modelo: {e}"), 500

    return jsonify(
        mensaje="Modelo reentrenado exitosamente",
        metricas={
            "precision": round(precision_macro, 4),
            "recall": round(recall_macro, 4),
            "f1": round(f1_macro, 4),
            "matriz_confusion": cm
        }
    ), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)