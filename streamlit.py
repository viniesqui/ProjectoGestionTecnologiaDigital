import streamlit as st
import joblib
import pandas as pd
from predict import ModelPredictor
import requests
import json
from joblib import load

model = joblib.load('model.joblib')

#hacerle los inputs
model_name = st.text_input("Modelo (en minuscula)")
year = st.number_input("Año", min_value=1900, max_value=2022, step=1)
milage = st.number_input("Millaje", min_value=0, step=1)
fuel = st.text_input("Tipo Gasolina(Diesel o Gasoline)")

# crear dataframe
input_data = pd.DataFrame({
    'model': [model_name], 
    'year': [year], 
    'mileage': [milage], 
    'fuel': [fuel]
})


if st.button('Predict'):
    # crear dataframe
    input_data = pd.DataFrame({
        'model': [model_name],
        'year': [year],
        'mileage': [milage],
        'fuel': [fuel]
    })

    # Create a new ModelPredictor instance
    predictor = ModelPredictor(model, input_data, columns_to_remove=[
    ], columns_to_convert=["model", "fuel"])

    # Use the predictor to make a prediction
    prediccion = predictor.predict()

    st.write(f"El precio estimado es: {prediccion[0]}, en Zloty (moneda polonia)")



# aca va la parte de la segunda version que es poder meter datos, la idea a futuro es que sirva como un 
#sistema de inventario para una empresa de carros
# Define the URL of your FastAPI application
url = 'http://localhost:8000/add_data/'

# Load the encoders
columns_to_convert = ['model', 'fuel'] 
encoders = {column: load(f'{column}_encoder.joblib') for column in columns_to_convert}

# Create a form to get the user's input
with st.form(key='my_form'):
    model = st.text_input(label='Enter model')
    year = st.number_input(label='Enter year', format='%d')
    mileage = st.number_input(label='Enter mileage', format='%d')
    fuel = st.text_input(label='Enter fuel')
    submit_button = st.form_submit_button(label='Submit')

# When the user clicks the submit button, transform the data and send a POST request to the FastAPI
import pandas as pd

if submit_button:
    # Create a DataFrame from the input data
    input_data = pd.DataFrame({
        'model': [model],
        'year': [year],
        'mileage': [mileage],
        'fuel': [fuel]
    })

    # Create a new ModelPredictor instance
    predictor = ModelPredictor(None, input_data, columns_to_convert=columns_to_convert)

    # Use the predictor to transform the data
    transformed_data = predictor.transform_data()

    # Convert the transformed data to a dictionary and then to a JSON string
    json_data = json.dumps(transformed_data.to_dict(orient='records')[0])

    response = requests.post(url, data=json_data)
    if response.status_code == 200:
        st.write('Data successfully added and model retrained if necessary.')
    else:
        st.write('An error occurred.')