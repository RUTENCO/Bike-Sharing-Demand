import os
import argparse
import pandas as pd
import joblib
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

def parse_args():
    """
    Define y parsea los argumentos de entrada para el script de entrenamiento.
    --input_file: CSV de entrenamiento (train.csv)
    --model_file: Ruta donde se guardará el modelo .pkl
    --scaler_file: Ruta donde se guardará el scaler .pkl
    --overwrite_model: Flag para sobrescribir modelo existente
    """
    parser = argparse.ArgumentParser(description="Entrena modelo LGBM para Bike Sharing Demand")
    parser.add_argument("--input_file",  required=True, help="CSV de entrenamiento (train.csv)")
    parser.add_argument("--model_file",  required=True, help="Archivo donde guardar el modelo .pkl")
    parser.add_argument("--scaler_file", required=True, help="Archivo donde guardar el scaler .pkl")
    parser.add_argument("--overwrite_model", action="store_true", help="Sobrescribe model_file si existe")
    return parser.parse_args()

def load_and_preprocess(input_path):
    """
    Carga train.csv, convierte 'datetime', extrae columnas temporales,
    elimina outliers en 'count' y devuelve DataFrame filtrado.
    """
    df = pd.read_csv(input_path, parse_dates=["datetime"])
    # Extraer características temporales
    df["hour"]  = df["datetime"].dt.hour
    df["day"]   = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["year"]  = df["datetime"].dt.year

    # Eliminar outliers en 'count' (>3σ)
    before = df.shape[0]
    df = df[abs(df["count"] - df["count"].mean()) < 3 * df["count"].std()]
    logger.info(f"Outliers eliminados: {before - df.shape[0]} filas")
    df.reset_index(drop=True, inplace=True)
    return df

def train_and_save(df, model_path, scaler_path):
    """
    Entrena un LGBMRegressor sobre df, guarda modelo y scaler.
    """
    # Preparar X e y
    X = df.drop(columns=["datetime", "casual", "registered", "count"])
    y = df["count"]

    # Escalar características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Entrenar LightGBM
    model = LGBMRegressor(n_estimators=500,
                          learning_rate=0.05,
                          subsample=0.8,
                          colsample_bytree=0.8,
                          random_state=42)
    logger.info("Entrenando modelo LGBM")
    model.fit(X_scaled, y)

    # Guardar modelo y scaler
    joblib.dump(model, scaler_path.replace(".pkl", "_model.pkl"))  # Para consistencia de nombres
    joblib.dump(scaler, scaler_path)
    logger.success(f"Modelo guardado en {model_path}")
    logger.success(f"Scaler guardado en {scaler_path}")

def main():
    args = parse_args()

    # Validaciones de archivos
    if not os.path.isfile(args.input_file):
        logger.error(f"No existe el archivo: {args.input_file}")
        exit(-1)
    if os.path.isfile(args.model_file) and not args.overwrite_model:
        logger.error(f"El modelo ya existe: {args.model_file}. Usa --overwrite_model")
        exit(-1)

    df = load_and_preprocess(args.input_file)
    train_and_save(df, args.model_file, args.scaler_file)

if __name__ == "__main__":
    main()
