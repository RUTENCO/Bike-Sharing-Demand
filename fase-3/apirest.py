
"""
API REST para el proyecto Bike Sharing Demand

Expone dos endpoints:
- POST /predict: recibe datos en JSON y devuelve predicciones de demanda de bicicletas.
- POST /train: entrena nuevamente el modelo usando el archivo CSV de entrenamiento.

Este servicio está diseñado para ejecutarse dentro de un contenedor Docker,
con los artefactos (modelo, scaler y CSV) montados en el volumen /data.
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

app = Flask(__name__)

# -------------------------------------------------------------------
# Rutas de los artefactos en el volumen /data (montado desde host)
MODEL_PATH  = "/data/model.pkl"
SCALER_PATH = "/data/scaler.pkl"
TRAIN_CSV   = "/data/train.csv"

# Cargar modelo y scaler una sola vez al iniciar el servidor
model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Lista de columnas EXACTA y en el mismo orden que usó el scaler al entrenar
FEATURE_COLUMNS = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "hour", "day", "month", "year"
]
# -------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint /predict
    Recibe JSON con datos de entrada. Ejemplo:
    {
      "season":     [1, 2],
      "holiday":    [0, 0],
      ...
      "windspeed":  [0.0, 3.2],
      "datetime":   ["2011-01-20 06:00:00", "2011-01-20 07:00:00"]
    }
    (la clave "datetime" es opcional si ya envías las columnas temporales)
    Devuelve JSON con las predicciones:
    { "count": [12,  8] }
    """
    data = request.get_json(force=True)
    df = pd.DataFrame(data)

    # Si se envía datetime, extraer 'hour','day','month','year'
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["hour"]  = df["datetime"].dt.hour
        df["day"]   = df["datetime"].dt.day
        df["month"] = df["datetime"].dt.month
        df["year"]  = df["datetime"].dt.year

    # Filtrar columnas y garantizar orden
    try:
        X = df[FEATURE_COLUMNS]
    except KeyError as e:
        missing = list(set(FEATURE_COLUMNS) - set(df.columns))
        return jsonify({
            "error": "Columnas faltantes o mal nombradas",
            "missing_columns": missing
        }), 400

    # Escalar y predecir
    X_scaled = scaler.transform(X)
    preds    = model.predict(X_scaled)
    preds    = np.maximum(0, preds).round().astype(int)

    return jsonify({"count": preds.tolist()})


@app.route("/train", methods=["POST"])
def train():
    """
    Endpoint /train
    Reentrena el modelo con el CSV de entrenamiento en /data/train.csv.
    Actualiza model.pkl y scaler.pkl en /data.
    Devuelve mensaje de éxito.
    """
    # 1. Leer datos
    df = pd.read_csv(TRAIN_CSV, parse_dates=["datetime"])

    # 2. Extraer variables temporales
    df["hour"]  = df["datetime"].dt.hour
    df["day"]   = df["datetime"].dt.day
    df["month"] = df["datetime"].dt.month
    df["year"]  = df["datetime"].dt.year

    # 3. Eliminar outliers en 'count'
    y = df["count"]
    df = df[abs(y - y.mean()) < 3 * y.std()]
    df.reset_index(drop=True, inplace=True)

    # 4. Preparar X e y
    X = df.drop(columns=["datetime", "casual", "registered", "count"])
    y = df["count"]

    # 5. Escalado
    new_scaler = StandardScaler()
    X_scaled   = new_scaler.fit_transform(X)

    # 6. Entrenar LightGBM
    new_model = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    new_model.fit(X_scaled, y)

    # 7. Guardar artefactos
    joblib.dump(new_model, MODEL_PATH)
    joblib.dump(new_scaler, SCALER_PATH)

    return jsonify({"message": "Modelo y scaler actualizados correctamente."})


if __name__ == "__main__":
    # Levanta el servidor Flask en el puerto 5000 aceptando conexiones externas
    app.run(host="0.0.0.0", port=5000)
