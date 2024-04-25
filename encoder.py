import pandas as pd

class CustomEncoder:
    def __init__(self):
        self.mapping = {}

    def fit_transform(self, data):
        unique_values = pd.unique(data)
        self.mapping = {value: i for i, value in enumerate(unique_values)}
        return data.map(self.mapping)

    def transform(self, data):
        return data.map(self.mapping)

    def inverse_transform(self, data):
        inverse_mapping = {v: k for k, v in self.mapping.items()}
        return data.map(inverse_mapping)