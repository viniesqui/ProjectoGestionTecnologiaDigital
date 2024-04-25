import pandas as pd
from sklearn.preprocessing import LabelEncoder


class ReadCsv:
    def __init__(self, data_filepath):
        self.data_filepath = data_filepath

    def read_data(self):
        return pd.read_csv(self.data_filepath)

class DataPreprocessor:
    def __init__(self, data, columns_to_remove=[], columns_to_convert=[]):
        self.df = data #quitamos la dependencia de un archivo csv, ahora se da el dataframe
        self.encoders = {}  # se le mete esto para hacerle el inverso
        self.remove_columns(columns_to_remove)
        self.transform_columns_to_numeric(columns_to_convert)

    def remove_columns(self, columns):
        self.df = self.df.drop(columns, axis=1)

    def transform_columns_to_numeric(self, columns):
        for column in columns:
            le = LabelEncoder()
            self.df[column] = le.fit_transform(self.df[column])
            self.encoders[column] = le 

    def get_data(self):
        return self.df

    def get_encoders(self):
        return self.encoders