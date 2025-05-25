# Dockerfile
FROM python:3.8-slim

WORKDIR /app

# Instalar dependencia para LightGBM
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Instalar librerías Python necesarias
RUN pip install --no-cache-dir \
    pandas numpy scikit-learn lightgbm joblib loguru

# Copiar scripts al contenedor
COPY train.py predict.py ./

# Definir punto de montaje para datos
VOLUME ["/data"]

# Por defecto, el contenedor correrá Python (ENTRYPOINT)
ENTRYPOINT ["python"]
