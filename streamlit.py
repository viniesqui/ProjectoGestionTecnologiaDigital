import joblib
import streamlit as st
import pandas as pd
from preprocessing import InputPreprocessor

model = joblib.load('model.joblib')

def predict_price(model_name, year, milage, fuel):
    data = {'model': [model_name],
            'year': [year],
            'mileage': [milage],
            'fuel': [fuel]}
    df = pd.DataFrame(data)
    print(df, "before")
    preprocessor = InputPreprocessor(df, columns_to_convert=['model', 'fuel'])
    df = preprocessor.preprocess_data()
    print(df, "after")
    print(f"Preprocessed data shape: {df.shape}")
    print(f"Model expected input shape: {model.n_features_in_}")

    # Check number of features
    if df.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Input data has {df.shape[1]} features, but model expects {model.n_features_in_} features")

    # Make prediction
    prediction = model.predict(df)
    return prediction[0]
model_name = st.text_input("Modelo")
year = st.number_input("Año", min_value=1900, max_value=2022, step=1)
milage = st.number_input("Millaje", min_value=0, step=1)
fuel = st.text_input("Tipo Gasolina")

if st.button('Predict'):
    predicted_price = predict_price(model_name, year, milage, fuel)
    st.write(f'Predicted Price: {predicted_price}')