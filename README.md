## 🚲 Bike Sharing Demand - Kaggle Challenge

### 📌 Descripción del reto
Los sistemas de bicicletas compartidas permiten alquilar bicicletas por periodos cortos a través de estaciones automáticas distribuidas por la ciudad. En este desafío, se pide predecir cuántas bicicletas serán alquiladas en un determinado momento, utilizando información sobre el clima, la fecha y otros factores.

El dataset contiene información horaria de dos años, y el entrenamiento se realiza con los primeros 19 días de cada mes, mientras que las predicciones deben hacerse desde el día 20 en adelante.

La métrica de evaluación utilizada es el Root Mean Squared Logarithmic Error (RMSLE).

---

### 📦 Objetivo

Predecir el número total de alquileres (`count`) por hora, utilizando únicamente la información disponible **previa al momento de la predicción**.

---

### 📁 Estructura del Dataset

| Columna      | Descripción |
|--------------|-------------|
| `datetime`   | Fecha y hora del registro |
| `season`     | Temporada del año (1: primavera, 2: verano, 3: otoño, 4: invierno) |
| `holiday`    | Si el día fue festivo |
| `workingday` | Si fue un día laboral (ni fin de semana ni festivo) |
| `weather`    | Condición climática categorizada (de 1 a 4) |
| `temp`       | Temperatura en °C |
| `atemp`      | Sensación térmica en °C |
| `humidity`   | Humedad relativa |
| `windspeed`  | Velocidad del viento |
| `casual`     | Alquileres por usuarios no registrados |
| `registered` | Alquileres por usuarios registrados |
| `count`      | Total de alquileres por hora (objetivo de predicción) |

---

### 🧪 Evaluación

Las predicciones se evalúan con la métrica **RMSLE (Root Mean Squared Logarithmic Error)**, que penaliza más las predicciones por debajo del valor real y es adecuada para distribuciones sesgadas con valores extremos.

![Captura de la formula RMSLE](RMSLE.png)

Donde:
- \( p_i \): valor predicho
- \( a_i \): valor real
- \( n \): número de muestras

---

### 🔗 Fuente del dataset

El conjunto de datos fue provisto por **Hadi Fanaee Tork** y es parte del repositorio de aprendizaje automático de la UCI. Se utiliza principalmente para fines educativos y de práctica en modelado predictivo.

---

### 🧪 Fase 1 - Exploración y modelado

En esta fase se realiza:

Carga de los datos y verificación de valores faltantes.

Análisis exploratorio de variables (EDA).

Ingeniería de características: extracción de hora, mes, día, año desde datetime.

Visualización de correlaciones y comportamiento de la variable objetivo.

Entrenamiento de modelos base:

- 🎯 Random Forest

- ⚡ XGBoost

- 🌿 LightGBM

Generación de archivos submission.csv para subir a Kaggle.
