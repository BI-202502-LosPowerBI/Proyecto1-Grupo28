from flask import Flask, request, jsonify
import os, joblib, numpy as np

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "modelo_ods.joblib")

# Se carga el Pipeline entrenado
try:
    model = joblib.load(MODEL_PATH)
    print(f"Modelo cargado: {MODEL_PATH}")
except Exception as e:
    print(f"Error cargando modelo en {MODEL_PATH}: {e}")
    model = None

@app.get("/ping")
def ping():
    return jsonify(ok=True)

@app.post("/predict")
def predict():
    if model is None:
        return jsonify(error="Modelo no cargado"), 500
    data = request.get_json(force=True)
    if "instancias" not in data:
        return jsonify(error="Formato inválido. Se requiere 'instancias'"), 400
    textos = [it["texto"] for it in data["instancias"]]
    preds = model.predict(textos).tolist()
    probs = None
    try:
        proba = model.predict_proba(textos)
        idx = np.argmax(proba, axis=1)
        probs = [float(proba[i, idx[i]]) for i in range(len(textos))]
    except Exception:
        pass
    return jsonify(predicciones=preds, probabilidades=probs)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
