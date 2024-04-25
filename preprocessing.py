import pandas as pd
from sklearn.preprocessing import LabelEncoder
from encoder import CustomEncoder
import joblib
from sklearn.impute import SimpleImputer

class ReadCsv:
    def __init__(self, data_filepath):
        self.data_filepath = data_filepath

    def read_data(self):
        return pd.read_csv(self.data_filepath)

    def preprocess_and_save_data(self, output_filepath, columns_to_remove, columns_to_convert):
        df = self.read_data()

        preprocessor = DataPreprocessor(
            df, columns_to_remove, columns_to_convert)
        preprocessed_data = preprocessor.get_data()

        preprocessed_data.to_csv(output_filepath, index=False)

        # Save the fitted encoders
        encoders = preprocessor.get_encoders()
        for column, encoder in encoders.items():
            joblib.dump(encoder, f'{column}_encoder.joblib')
        return preprocessor.get_encoders()


class DataPreprocessor:
    def __init__(self, data, columns_to_remove=[], columns_to_convert=[]):
        self.df = data  # quitamos la dependencia de un archivo csv, ahora se da el dataframe
        self.encoders = {}  # se le mete esto para hacerle el inverso
        self.remove_columns(columns_to_remove)
        self.transform_columns_to_numeric(columns_to_convert)

    def remove_columns(self, columns):
        self.df = self.df.drop(columns, axis=1)

    def transform_columns_to_numeric(self, columns):
        for column in columns:
            ce = CustomEncoder()  # Use your CustomEncoder
            self.df[column] = ce.fit_transform(self.df[column])
            self.encoders[column] = ce

        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        self.df = pd.DataFrame(imputer.fit_transform(
            self.df), columns=self.df.columns)

    def get_data(self):
        return self.df

    def get_encoders(self):
        return self.encoders




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
            ce = CustomEncoder()  # Use your CustomEncoder
            self.df[column] = ce.fit_transform(self.df[column])
            self.encoders[column] = ce

    def get_data(self):
        return self.df

    def get_encoders(self):
        return self.encoders
    
    
