from encoder import CustomEncoder
from joblib import load, dump
import pandas as pd

class ModelPredictor:
    def __init__(self, model, data, columns_to_remove=[], columns_to_convert=[]):
        if isinstance(model, str):
            self.model = load(model)
        else:
            self.model = model
        self.data = data
        self.columns_to_remove = columns_to_remove
        self.columns_to_convert = columns_to_convert
        self.encoders = {column: load(f'{column}_encoder.joblib') for column in columns_to_convert}
        print("Encoders loaded: ", self.encoders)  # Print the loaded encoders

    def predict(self):
        if isinstance(self.data, pd.DataFrame):
            # Transform the specified columns to numeric
            for column in self.columns_to_convert:
                print("Before transformation: ", self.data[column])  # Print the column data before transformation
                self.data[column] = self.encoders[column].transform(self.data[column])
                print("After transformation: ", self.data[column])  # Print the column data after transformation
            print("Data after transformation: ", self.data)
            prediction = self.model.predict(self.data)
            print("Prediction: ", prediction)  # Print the prediction
            return prediction
        else:
            raise TypeError("Input data should be a pandas DataFrame")