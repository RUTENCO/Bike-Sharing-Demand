import requests

BASE_URL = "http://localhost:5000"

def test_predict():
    # Ejemplo con 2 filas de input
    payload = {
        "hour":  [6, 7],
        "day":   [20, 20],
        "month": [1, 1],
        "year":  [2011, 2011],
        "season":     [1, 1],
        "holiday":    [0, 0],
        "workingday": [0, 0],
        "weather":    [1, 1],
        "temp":       [9.84, 9.02],
        "atemp":      [14.395, 13.635],
        "humidity":   [81, 80],
        "windspeed":  [0.0, 0.0]
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    print("Predict response:", resp.json())

def test_train():
    resp = requests.post(f"{BASE_URL}/train")
    print("Train response:", resp.json())

if __name__ == "__main__":
    test_predict()
    test_train()
