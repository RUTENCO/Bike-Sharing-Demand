import os
import argparse
import pandas as pd
import numpy as np
import joblib
from loguru import logger

def parse_args():
    """
    Define y parsea los argumentos de entrada para el script de predicción.
    --input_file: CSV de test (test.csv)
    --model_file: Ruta al modelo entrenado .pkl
    --scaler_file: Ruta al scaler .pkl
    --predictions_file: Archivo de salida submission.csv
    """
    parser = argparse.ArgumentParser(description="Predice demanda de Bike Sharing")
    parser.add_argument("--input_file",       required=True, help="CSV de prueba (test.csv)")
    parser.add_argument("--model_file",       required=True, help="Modelo entrenado (.pkl)")
    parser.add_argument("--scaler_file",      required=True, help="Scaler usado (.pkl)")
    parser.add_argument("--predictions_file", required=True, help="Salida CSV (submission.csv)")
    return parser.parse_args()

def load_and_transform(input_path, scaler_path):
    """
    Carga test.csv, parsea 'datetime', extrae variables temporales,
    aplica scaler guardado y devuelve DataFrame y matriz escalada.
    """
    df = pd.read_csv(input_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["hour"]  = df["datetime"].dt.hour
    df["day"]   = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["year"]  = df["datetime"].dt.year

    X_test = df.drop(columns=["datetime"])
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X_test)
    return df, X_scaled

def predict_and_save(df, X_scaled, model_path, output_path):
    """
    Carga modelo entrenado, genera predicciones no negativas,
    crea submission.csv con 'datetime,count' y lo guarda.
    """
    model = joblib.load(model_path)
    preds = model.predict(X_scaled)
    preds = np.maximum(0, preds).round().astype(int)

    submission = pd.DataFrame({
        "datetime": df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "count":    preds
    })
    submission.to_csv(output_path, index=False)
    logger.success(f"Predicciones guardadas en: {output_path}")

def main():
    args = parse_args()

    # Validaciones
    if not os.path.isfile(args.input_file):
        logger.error(f"No existe test: {args.input_file}")
        exit(-1)
    if not os.path.isfile(args.model_file) or not os.path.isfile(args.scaler_file):
        logger.error("Faltan modelo o scaler")
        exit(-1)

    df, X_scaled = load_and_transform(args.input_file, args.scaler_file)
    predict_and_save(df, X_scaled, args.model_file, args.predictions_file)

if __name__ == "__main__":
    main()
