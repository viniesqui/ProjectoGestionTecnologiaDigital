import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

class ModelTrainerRegression:
    def __init__(self, df, target_column, model_filepath):
        self.df = df
        self.target_column = target_column
        self.model_filepath = model_filepath

    def split_data(self):
        self.X = self.df.drop(self.target_column, axis=1)
        self.y = self.df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def train_model(self):
        model = LinearRegression()
        model.fit(self.X_train, self.y_train)
        return model

    def serialize_model(self, model):
        joblib.dump(model, self.model_filepath)

    def run(self):
        self.split_data()
        trained_model = self.train_model()
        self.serialize_model(trained_model)