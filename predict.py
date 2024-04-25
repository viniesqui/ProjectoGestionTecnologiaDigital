from joblib import load
import pandas as pd

class ModelPredictor:
    def __init__(self, model):
        self.model = model

    def load_model(self):
        self.model = load(self.model)

    def predict(self, input_data):
        if isinstance(input_data, pd.DataFrame):
            return self.model.predict(input_data)
        else:
            raise TypeError("Input data should be a pandas DataFrame")

