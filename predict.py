from joblib import load
import pandas as pd

class ModelPredictor:
    def __init__(self, model_filepath):
        self.model_filepath = model_filepath

    def load_model(self):
        self.model = load(self.model_filepath)

    def predict(self, input_data):
        if isinstance(input_data, pd.DataFrame):
            return self.model.predict(input_data)
        else:
            raise TypeError("Input data should be a pandas DataFrame")

