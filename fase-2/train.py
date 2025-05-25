# train.py
import os
import argparse
import pandas as pd
import joblib
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

def main():
    # 1. Argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Entrena modelo LGBM para Bike Sharing Demand")
    parser.add_argument("--input_file",  required=True, help="CSV de entrenamiento (train.csv)")
    parser.add_argument("--model_file",  required=True, help="Archivo donde se guardará el modelo")
    parser.add_argument("--scaler_file", required=True, help="Archivo donde se guardará el scaler")
    parser.add_argument("--overwrite_model", action="store_true",
                        help="Permite sobrescribir si el modelo ya existe")
    args = parser.parse_args()

    # 2. Validar archivos de entrada y existencia de modelo
    if not os.path.isfile(args.input_file):
        logger.error(f"No existe el archivo: {args.input_file}")
        exit(-1)
    if os.path.isfile(args.model_file) and not args.overwrite_model:
        logger.error(f"El modelo ya existe: {args.model_file}. Usa --overwrite_model para sobrescribir.")
        exit(-1)

    # 3. Cargar datos y parsear datetime
    logger.info("Cargando datos de entrenamiento")
    df = pd.read_csv(args.input_file, parse_dates=["datetime"])

    # 4. Crear variables de tiempo
    df["hour"]  = df["datetime"].dt.hour
    df["day"]   = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["year"]  = df["datetime"].dt.year

    # 5. Eliminar outliers en 'count' (>3σ)
    before = df.shape[0]
    df = df[abs(df["count"] - df["count"].mean()) < 3 * df["count"].std()]
    logger.info(f"Outliers eliminados: {before - df.shape[0]} filas")
    df.reset_index(drop=True, inplace=True)

    # 6. Preparar X e y
    X = df.drop(columns=["datetime", "casual", "registered", "count"])
    y = df["count"]

    # 7. Escalar características
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 8. Entrenar modelo LightGBM
    model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    logger.info("Entrenando modelo LGBM")
    model.fit(X_scaled, y)

    # 9. Guardar modelo y scaler a disco
    joblib.dump(model, args.model_file)
    joblib.dump(scaler, args.scaler_file)
    logger.success(f"✔ Modelo guardado en {args.model_file}")
    logger.success(f"✔ Scaler guardado en {args.scaler_file}")

if __name__ == "__main__":
    main()
