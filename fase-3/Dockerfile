# Partimos de la imagen bikeshare creada en fase-2
FROM bikeshare

# Instala Flask y requests necesarios para API y cliente
RUN pip install --no-cache-dir flask requests

# Copia los scripts de la fase-3
COPY apirest.py client.py ./

# Expone el puerto 5000
EXPOSE 5000

# Por defecto, arranca el servidor Flask
ENTRYPOINT ["python", "apirest.py"]
