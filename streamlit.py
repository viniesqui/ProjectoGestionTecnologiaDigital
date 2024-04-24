from joblib import load
import streamlit as st
import pandas as pd
from preprocessing import InputPreprocessor

model = load('preprocessor.joblib')


model_name = st.text_input("Modelo")
year = st.number_input("Año", min_value=1900, max_value=2022, step=1)
milage = st.number_input("Millaje", min_value=0, step=1)
fuel = st.text_input("Tipo Gasolina")


if st.button('Predict'):
    data = {'model': [model_name],
        'year': [year],
        'mileage': [milage],
        'fuel': [fuel]}
    df = pd.DataFrame(data)
    
    
    preprocessor = InputPreprocessor(df, columns_to_convert=['model', 'fuel'])
    df = preprocessor.preprocess_data()
    
    
    prediction = model.predict(df)
    
   
    st.write(f'Predicted Price: {prediction[0]}')