📝 Descripción del Reto - Bike Sharing Demand (Kaggle)
El desafío “Bike Sharing Demand” de Kaggle consiste en predecir la demanda de alquiler de bicicletas por hora en el sistema Capital Bikeshare de Washington D.C., combinando datos históricos de uso con variables climáticas y temporales.

Los sistemas de bicicletas compartidas generan grandes volúmenes de datos al registrar el inicio, duración y ubicación de los viajes. Esta competencia plantea el reto de modelar dichos patrones para anticipar cuántas bicicletas serán alquiladas en cada hora del día.

📦 Objetivo
Predecir el número total de alquileres (count) por hora, utilizando únicamente la información disponible previa al momento de la predicción.

📁 Estructura del Dataset
Columnas principales:
datetime: Fecha y hora del registro

season: Temporada del año (1: primavera, 2: verano, 3: otoño, 4: invierno)

holiday: Si el día fue festivo

workingday: Si fue un día laboral (ni fin de semana ni festivo)

weather: Condición climática categorizada

temp: Temperatura en °C

atemp: Sensación térmica en °C

humidity: Humedad relativa

windspeed: Velocidad del viento

casual: Alquileres por usuarios no registrados

registered: Alquileres por usuarios registrados

count: Total de alquileres por hora (objetivo de predicción)

🧪 Evaluación
Las predicciones se evalúan con la métrica RMSLE (Root Mean Squared Logarithmic Error), que penaliza más las predicciones por debajo del valor real y es adecuada para distribuciones sesgadas con valores extremos.


🔗 Fuente del dataset
El conjunto de datos fue provisto por Hadi Fanaee Tork y es parte del repositorio de aprendizaje automático de la UCI. Se utiliza principalmente para fines educativos y de práctica en modelado predictivo.
