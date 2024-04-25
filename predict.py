from joblib import load
import pandas as pd
from preprocessing import DataPreprocessor

class ModelPredictor:
    def __init__(self, model, data, columns_to_remove=[], columns_to_convert=[]):
        self.model = model
        self.preprocessor = DataPreprocessor(
            data, columns_to_remove=[], columns_to_convert=[])

    def load_model(self):
        self.model = load(self.model)

    def predict(self):
        if isinstance(self.data, pd.DataFrame):
            # Transform the columns to numeric
            self.preprocessor.df = self.data
            self.preprocessor.transform_columns_to_numeric(self.data.columns)
            preprocessed_data = self.preprocessor.df

            return self.model.predict(preprocessed_data)
        else:
            raise TypeError("Input data should be a pandas DataFrame")