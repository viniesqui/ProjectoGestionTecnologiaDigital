import streamlit as st
import joblib
import pandas as pd
from predict import ModelPredictor

# cargar el modelo
model = joblib.load('model.joblib')


#hacerle los inputs
model_name = st.text_input("Modelo")
year = st.number_input("Año", min_value=1900, max_value=2022, step=1)
milage = st.number_input("Millaje", min_value=0, step=1)
fuel = st.text_input("Tipo Gasolina")

# crear dataframe
input_data = pd.DataFrame({
    'model': [model_name], 
    'year': [year], 
    'mileage': [milage], 
    'fuel': [fuel]
})

# Create a new ModelPredictor instance
predictor = ModelPredictor(model, input_data, columns_to_remove=[], columns_to_convert=["model", "fuel"])

# Use the predictor to make a prediction
prediccion = predictor.predict()


st.write(f"El precio estimado es: {prediccion[0]}")