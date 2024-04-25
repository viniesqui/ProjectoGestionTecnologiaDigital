import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

class CustomLabelEncoder(LabelEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def inverse_transform(self, X):
        return super().inverse_transform(X)

class DataPreprocessor:
    def __init__(self, filepath, columns_to_remove=[], columns_to_convert=[]):
        self.df = pd.read_csv(filepath)
        self.remove_columns(columns_to_remove)
        self.transform_columns_to_numeric(columns_to_convert)

    def remove_columns(self, columns):
        self.df = self.df.drop(columns, axis=1)

    def transform_columns_to_numeric(self, columns):
        le = CustomLabelEncoder()
        for column in columns:
            self.df[column] = le.fit_transform(self.df[column])

    def preprocess_data(self):
        numeric_features = self.df.select_dtypes(
            include=['int64', 'float64']).columns
        categorical_features = self.df.select_dtypes(
            include=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('encoder', OrdinalEncoder())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        preprocessed_data = preprocessor.fit_transform(self.df)

        return preprocessed_data
        
        
class InputPreprocessor:
    def __init__(self, data):
        self.df = data

    def transform_columns_to_numeric(self, columns):
        for column in columns:
            self._encode_column(column)

    def _encode_column(self, column):
        le = LabelEncoder()
        self.df[column] = le.fit_transform(self.df[column])

    def preprocess_data(self):
        numeric_features = self._get_numeric_features()
        preprocessed_data = self._apply_preprocessing(numeric_features)
        return preprocessed_data

    def _get_numeric_features(self):
        numeric_features = self.df.columns
        numeric_features = [feature for feature in numeric_features if self.df[feature].nunique() > 1]
        return numeric_features

    def _apply_preprocessing(self, numeric_features):
        if not numeric_features:
            return self.df

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features)
            ])

        preprocessed_data = preprocessor.fit_transform(self.df)
        return preprocessed_data