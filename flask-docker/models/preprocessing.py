import pandas as pd

def limpieza_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza limpieza básica del DataFrame:
    - Elimina filas duplicadas
    - Elimina filas con valores nulos
    - Elimina filas con textos vacíos
    - Convierte columnas a tipos adecuados

    Args:
        df (pd.DataFrame): DataFrame original

    Returns:
        pd.DataFrame: DataFrame limpio
    """
    # Eliminar filas duplicadas
    df = df.drop_duplicates().reset_index(drop=True)

    # Eliminar filas con valores nulos
    df = df.dropna().reset_index(drop=True)

    # Convertir columnas a tipos adecuados
    df["textos"] = df["textos"].astype("string")
    df["labels"] = df["labels"].astype("int64")

    # Eliminar filas con textos vacíos
    texto_vacio = df["textos"].astype("string").str.strip().eq("")
    df = df[~texto_vacio].reset_index(drop=True)

    # Eliminar filas con textos duplicados
    df = df.drop_duplicates(subset=["textos"], keep="first").reset_index(drop=True)

    return df