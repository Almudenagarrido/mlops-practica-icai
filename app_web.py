import streamlit as st
import requests
import json
import os
import numpy as np

# Titulo de la aplicacion
st.title('API de Prediccion del Modelo Iris')

# Descripcion de la aplicacion
st.write('Ingresa las caracteristicas de la flor de iris para obtener una prediccion de su especie.')

# Campos de entrada para las caracteristicas del iris
sepal_length = st.slider('Longitud del sepalo (cm)', 0.0, 10.0, 5.0)
sepal_width = st.slider('Ancho del sepalo (cm)', 0.0, 10.0, 3.0)
petal_length = st.slider('Longitud del petalo (cm)', 0.0, 10.0, 4.0)
petal_width = st.slider('Ancho del petalo (cm)', 0.0, 10.0, 1.0)

# Boton para hacer la prediccion
if st.button('Obtener Prediccion'):
    # Crear los datos en formato JSON
    features = [sepal_length, sepal_width, petal_length, petal_width]
    payload = {'features': features}

    # URL de la API de prediccion (cambia esta URL si la API no esta en esa url)
    api_url = os.environ.get('API_URL')

    try:
        # Enviar la peticion a la API
        response = requests.post(api_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})

        # Verificar si la peticion fue exitosa
        if response.status_code == 200:
            prediction_result = response.json().get('prediction')

            # Mapear el resultado numerico a una especie
            species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
            predicted_species = species_map.get(prediction_result, 'Desconocida')

            st.success(f"La prediccion es: **{predicted_species}**")
        else:
            st.error(f"Error en la peticion: {response.status_code} - {response.text}")
    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo conectar con la API. Asegurate de que esta en ejecucion. Error: {e}")
