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

### 🧠 Fase 1 - Exploración y Modelado con Notebook

#### 🔧 Requisitos previos

Puedes ejecutar este proyecto directamente en Google Colab. Para ello:

Descarga el notebook **(EDA)_Bike_Sharing_Demand.ipynb**.

Sube el archivo a tu Google Drive.

Ábrelo desde Colab: haz clic derecho sobre el archivo en Drive → Abrir con → Google Colab.

**Nota:**  
La solución presentada en este notebook se basa en trabajos previos de la comunidad de Kaggle.  
En particular, se han tomado ideas y enfoques de los siguientes notebooks:

- 📘 [**biketest** por ensari](https://www.kaggle.com/code/ensari/biketest)
- 📗 [**Bike Sharing Demand** por jaeraklim](https://www.kaggle.com/code/jaeraklim/bike-sharing-demand)

Estas soluciones sirvieron como guía para realizar la ingeniería de características, entrenar modelos como Random Forest, XGBoost y LightGBM, y estructurar la lógica de predicción para el envío de resultados a la competencia.


📝 **Pasos para ejecutar el notebook**

1. Únete al reto en Kaggle y acepta las reglas para habilitar la descarga de los datos:

   👉 https://www.kaggle.com/competitions/bike-sharing-demand/data

2. Asegúrate de tener los siguientes archivos descargados desde Kaggle:

   - `train.csv`
   - `test.csv`

3. Existen dos formas de acceder a estos archivos desde el notebook en Google Colab:

   - **Opción A**: Subir los archivos directamente al entorno de Colab:
     - Haz clic en el ícono de carpeta (a la izquierda en Colab).
     - Luego haz clic en el ícono de subir archivo (flecha hacia arriba).
     - Carga `train.csv` y `test.csv`.

   - **Opción B**: Cargar archivos desde tu Google Drive:
     - Conecta Colab a tu Google Drive al ejecutar este código en una celda:
       ```python
       from google.colab import drive
       drive.mount('/content/drive')
       ```
     - Coloca los archivos en una carpeta de tu Drive (por ejemplo, `/content/drive/MyDrive/bike-sharing/`)
     - Luego carga los datos desde esa ruta en tu notebook.

Ejecuta el notebook (EDA)_Bike_Sharing_Demand.ipynb siguiendo las celdas paso a paso.

#### ⚙️ ¿Qué hace el notebook?

Carga de datos (train.csv, test.csv)

Limpieza y verificación de valores faltantes

Análisis exploratorio con visualizaciones de tendencias

Ingeniería de características:

Extracción de hour, day, month, year desde la columna datetime

Conversión de variables si es necesario

Entrenamiento de modelos:

🎲 Random Forest 

💡 LightGBM 

⚡ XGBoost 

Evaluación con la métrica **RMSLE**

Generación de archivos de envío (submission.csv) listos para subir a Kaggle

#### 📂 Archivos generados
**submission_rf.csv** → predicciones usando Random Forest

**submission_lgb.csv** → predicciones usando LightGBM

**submission_xgb.csv** → predicciones usando XGBoost

#### 🚀 Cómo subir a Kaggle
Ve al apartado "Submit Predictions" del reto:

👉 [https://www.kaggle.com/competitions/bike-sharing-demand/submit](https://www.kaggle.com/competitions/bike-sharing-demand/overview)

Carga alguno de los archivos .csv generados (por ejemplo, submission_lgb.csv).

Asigna un nombre a tu envío.

Haz clic en Make Submission.

Kaggle calculará la puntuación basada en la métrica RMSLE.
