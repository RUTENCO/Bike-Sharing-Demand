# Imagen base ligera con Python 3.8
FROM python:3.8-slim

# Directorio de trabajo en el contenedor
WORKDIR /app

# Instala libgomp (requisito de LightGBM) y limpia caché de apt
RUN apt-get update && \
    apt-get install -y libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Instala dependencias de Python necesarias
RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    lightgbm \
    joblib \
    loguru

# Copia los scripts de entrenamiento y predicción
COPY train.py predict.py ./

# Directorio donde montaremos 'train.csv', 'test.csv' y donde escribiremos los outputs
VOLUME ["/data"]

# Por defecto, al invocar el contenedor corre Python (se le pasan luego los scripts)
ENTRYPOINT ["python"]
