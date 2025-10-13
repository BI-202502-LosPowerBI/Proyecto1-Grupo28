import os, random, pandas as pd, warnings
from collections import Counter
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    from openai import OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Error: la variable de entorno OPENAI_API_KEY no esta configurada.")
    client = OpenAI()
    LLM_AVAILABLE = True
except Exception as e:
    print("No se puede inicializar OpenAI:", e)
    client = None
    LLM_AVAILABLE = False

# -------- Parámetros --------
SEED = 42
random.seed(SEED)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
DATA_DIR = os.path.join(BASE_DIR, "datos_etapa_2")  

ORIG_XLSX = os.path.join(DATA_DIR, "datos_proyecto.xlsx")
OUTPUT_CSV = os.path.join(DATA_DIR, "datos_proyecto_aumentados.csv")

TEXT_COL = "textos"
LABEL_COL = "labels"

TARGET_BALANCE_RATIO = 0.8
LLM_MODEL = "gpt-4o-mini"

def load_data_excel(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    df = pd.read_excel(path)
    assert {TEXT_COL, LABEL_COL}.issubset(df.columns), \
        f"El archivo {path} debe tener columnas '{TEXT_COL}' y '{LABEL_COL}'"
    df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df.dropna(subset=[LABEL_COL])
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    print(f"\n--- Diagnóstico de clases en {os.path.basename(path)} ---")
    print(Counter(df[LABEL_COL]))
    print("-----------------------------------------\n")
    return df

def llm_augment(sent, label, model=LLM_MODEL, n=3):
    """Genera ejemplos sintéticos mediante prompting con un modelo LLM de OpenAI."""
    if not LLM_AVAILABLE or client is None:
        print("No hay conexion a OpenAI")
        return []

    prompt = (
        f"Eres un asistente que genera ejemplos de texto para entrenamiento de un modelo de clasificación.\n"
        f"Clase objetivo: {label}\n"
        f"Texto base: \"{sent}\"\n"
        f"Devuelve {n} frases nuevas que mantengan el mismo tipo de sentimiento o intención, "
        f"sin numerarlas, en lenguaje natural y conciso."
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9
        )
        text = resp.choices[0].message.content
        lines = [t.strip("-• ").strip() for t in text.split("\n") if len(t.strip()) > 5]
        return lines[:n]
    except Exception as e:
        print("Error al generar ejemplos con LLM:", e)
        return []

def augment_minority(df, text_col=TEXT_COL, label_col=LABEL_COL, ratio=TARGET_BALANCE_RATIO):
    """Aumenta la clase minoritaria usando únicamente prompting con OpenAI."""
    c = Counter(df[label_col])
    if len(c) < 2:
        print("Solo hay una clase en el dataset, no se puede aumentar.")
        return df

    majority_label = max(c, key=c.get)
    minority_label = min(c, key=c.get)
    majority_count = c[majority_label]
    minority_count = c[minority_label]

    desired_minority = int(majority_count * ratio)
    to_create = max(0, desired_minority - minority_count)

    if to_create == 0:
        print("No se requiere aumentación (ya balanceado al ratio).")
        return df

    print(f"Mayoría: {majority_label}={majority_count}, minoría: {minority_label}={minority_count}.")
    print(f"Creando {to_create} ejemplos sintéticos mediante prompting con {LLM_MODEL}...")

    minority_texts = df[df[label_col] == minority_label][text_col].tolist()
    synthetic = []
    pbar = tqdm(total=to_create)

    while len(synthetic) < to_create:
        base = random.choice(minority_texts)
        created = llm_augment(base, minority_label, n=3)
        for t in created:
            if t and len(t.strip()) > 5:
                synthetic.append({text_col: t, label_col: minority_label})
                pbar.update(1)
                if len(synthetic) >= to_create:
                    break
    pbar.close()

    df_synth = pd.DataFrame(synthetic)
    df_aug = pd.concat([df, df_synth], ignore_index=True).sample(frac=1, random_state=SEED)
    print("Conteo post-aumentación:", Counter(df_aug[label_col]))
    return df_aug

# ---------- Main ----------
def main():
    print(f"\nCargando datos desde: {ORIG_XLSX}")
    df = load_data_excel(ORIG_XLSX)

    if not LLM_AVAILABLE:
        print("\n⚠️ Advertencia: no hay conexión a OpenAI. No se podrá hacer aumentación con prompting.")
        return

    augmented = augment_minority(df)

    os.makedirs(DATA_DIR, exist_ok=True)
    augmented.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\nArchivo CSV guardado correctamente en:\n{OUTPUT_CSV}")

if __name__ == "__main__":
    main()
