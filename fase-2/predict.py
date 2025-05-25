# predict.py
import os
import argparse
import pandas as pd
import numpy as np
import joblib
from loguru import logger

def main():
    # 1. Definición de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Genera predicciones para Bike Sharing Demand")
    parser.add_argument("--input_file",       required=True, help="Ruta al CSV de test (test.csv)")
    parser.add_argument("--model_file",       required=True, help="Ruta al modelo entrenado (model.pkl)")
    parser.add_argument("--scaler_file",      required=True, help="Ruta al scaler usado (scaler.pkl)")
    parser.add_argument("--predictions_file", required=True, help="Ruta donde se guardará submission.csv")
    args = parser.parse_args()

    # 2. Validar que los archivos existen
    if not os.path.isfile(args.input_file):
        logger.error(f"Test no existe: {args.input_file}")
        exit(-1)
    if not os.path.isfile(args.model_file) or not os.path.isfile(args.scaler_file):
        logger.error("Modelo o scaler no encontrados")
        exit(-1)

    # 3. Cargar datos de test y parsear datetime
    logger.info("Cargando test y extrayendo características")
    df = pd.read_csv(args.input_file)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # 4. Ingeniería de características temporales
    df["hour"]  = df["datetime"].dt.hour
    df["day"]   = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["year"]  = df["datetime"].dt.year

    # 5. Preparar matriz de features
    X_test = df.drop(columns=["datetime"])

    # 6. Cargar scaler y transformar las features
    scaler   = joblib.load(args.scaler_file)
    X_scaled = scaler.transform(X_test)

    # 7. Cargar modelo y predecir
    model = joblib.load(args.model_file)
    preds = model.predict(X_scaled)

    # 8. Asegurar predicciones ≥ 0 y enteros
    preds = np.round(np.maximum(0, preds)).astype(int)

    # 9. Construir DataFrame de envío con formato datetime completo
    submission = pd.DataFrame({
        "datetime": df["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "count":    preds
    })

    # 10. Guardar CSV sin índice
    submission.to_csv(args.predictions_file, index=False)
    logger.success(f"✔ Predicciones guardadas en: {args.predictions_file}")

if __name__ == "__main__":
    main()