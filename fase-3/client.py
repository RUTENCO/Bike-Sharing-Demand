"""
client.py

Cliente de ejemplo para consumir la API REST de Bike Sharing Demand
Endpoints:
  - POST /predict: recibe un batch de datos y devuelve predicciones.
  - POST /train:  retrena el modelo con los datos estándar.
"""

import requests

# URL base donde corre el servidor Flask dentro de Docker
BASE_URL = "http://localhost:5000"

def test_predict():
    """
    Envía una petición POST al endpoint /predict con un ejemplo de payload
    que contiene dos registros. Imprime la respuesta JSON con el array de counts.
    """
    # Datos de entrada: lista de características, cada una con 2 valores
    payload = {
        "hour":       [6, 7],
        "day":        [20, 20],
        "month":      [1, 1],
        "year":       [2011, 2011],
        "season":     [1, 1],
        "holiday":    [0, 0],
        "workingday": [0, 0],
        "weather":    [1, 1],
        "temp":       [9.84, 9.02],
        "atemp":      [14.395, 13.635],
        "humidity":   [81, 80],
        "windspeed":  [0.0, 0.0]
    }

    # Realiza la llamada al API
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    # Imprime el JSON recibido: {"count": [..., ...]}
    print("Predict response:", resp.json())

def test_train():
    """
    Envía una petición POST al endpoint /train para reentrenar el modelo
    usando el CSV de entrenamiento. Imprime el mensaje de éxito.
    """
    resp = requests.post(f"{BASE_URL}/train")
    print("Train response:", resp.json())

if __name__ == "__main__":
    # Primero probamos predict, luego train
    test_predict()
    test_train()
