import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

class DataPreprocessor:
    def __init__(self, filepath, columns_to_remove=[], columns_to_convert=[]):
        self.df = pd.read_csv(filepath)
        self.remove_columns(columns_to_remove)
        self.transform_columns_to_numeric(columns_to_convert)

    def remove_columns(self, columns):
        self.df = self.df.drop(columns, axis=1)

    def transform_columns_to_numeric(self, columns):
        le = LabelEncoder()
        for column in columns:
            self.df[column] = le.fit_transform(self.df[column])

    def preprocess_data(self):
        numeric_features = self.df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = self.df.select_dtypes(include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return preprocessor.fit_transform(self.df)