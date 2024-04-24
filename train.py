from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error
from joblib import dump, load
from preprocessing import DataPreprocessor
from sklearn.model_selection import GridSearchCV
import pandas as pd

class ModelTrainerRegression:
    def __init__(self, data_filepath, target_column, columns_to_remove, columns_to_convert, model_filepath='model.joblib'):
        self.data_filepath = data_filepath
        self.target_column = target_column
        self.model_filepath = model_filepath
        self.columns_to_remove = columns_to_remove
        self.columns_to_convert = columns_to_convert
        self.preprocessor = DataPreprocessor(data_filepath, columns_to_remove=self.columns_to_remove, columns_to_convert=self.columns_to_convert)
    
    def load_data(self):
        self.df = pd.read_csv(self.data_filepath)
        self.X = self.df.drop(self.target_column, axis=1)
        self.y = self.df[self.target_column]

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

    def train_models(self, pipelines, param_grids):
        best_mse = float('inf')
        best_pipeline = None

        for name, pipeline in pipelines.items():
            model = GridSearchCV(pipeline, param_grids[name], cv=5, n_jobs=-1)
            model.fit(self.X_train, self.y_train)
            predictions = model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, predictions)
            print(f"{name} MSE: {mse}")

            if mse < best_mse:
                best_mse = mse
                best_pipeline = model

        return best_pipeline

    def serialize_model(self, model):
        dump(model, self.model_filepath)

    def run(self):
        self.load_data()
        self.split_data()

        pipelines = {
            'lr': Pipeline(steps=[('scaler', StandardScaler()), ('selector', SelectKBest(score_func=f_regression)), ('classifier', LinearRegression())]),
            'rf': Pipeline(steps=[('scaler', StandardScaler()), ('selector', SelectKBest(score_func=f_regression)), ('classifier', RandomForestRegressor())]),
            'svr': Pipeline(steps=[('scaler', StandardScaler()), ('selector', SelectKBest(score_func=f_regression)), ('classifier', SVR())])
        }

        param_grids = {
            'lr': {
                'selector__k': [1, 2, 3, 4]  # Adjusted k values
            },
            'rf': {
                'selector__k': [1, 2, 3, 4],  # Adjusted k values
                'classifier__n_estimators': [10, 50, 100],
                'classifier__max_depth': [None, 10, 20],
                'classifier__min_samples_split': [2, 5, 10]	
            },
            'svr': {
                'selector__k': [1, 2, 3, 4],  # Adjusted k values
                'classifier__C': [0.1, 1, 10],
                'classifier__gamma': ['scale', 'auto']
            }
        }

        best_model = self.train_models(pipelines, param_grids)
        print(self.X.columns, self.y.name)
        print(best_model.best_params_)

        self.serialize_model(best_model)