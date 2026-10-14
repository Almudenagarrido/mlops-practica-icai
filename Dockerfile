# Usar una imagen base de Python 3.11
FROM python:3.11-slim

# Establecer el directorio de trabajo en el contenedor
WORKDIR /app

# Copiar el archivo de requisitos e instalar las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar los scripts de la aplicacion y el modelo
COPY app.py .
COPY model.pkl .

# Exponer el puerto en el que se ejecutara la aplicacion
EXPOSE 5000

# Comando para ejecutar la aplicacion cuando se inicie el contenedor
CMD ["python", "app.py"]
