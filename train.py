from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from joblib import dump
from preprocessing import DataPreprocessor
from sklearn.model_selection import GridSearchCV

class ModelTrainerRegression:
    def __init__(self, data_filepath, target_column, model_filepath):
        self.data_filepath = data_filepath
        self.target_column = target_column
        self.model_filepath = model_filepath
        self.preprocessor = DataPreprocessor(data_filepath)

    def train_models(self, pipelines, param_grids):
        best_mse = float('inf')
        best_pipeline = None

        for name, pipeline in pipelines.items():
            model = GridSearchCV(pipeline, param_grids[name], cv=5)
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
        # Preprocess the data
        df = self.preprocessor.df
        processed_data = self.preprocessor.preprocess_data()

        # Define the model pipeline
        pipelines = {
            'lr': Pipeline(steps=[('preprocessor', processed_data), ('classifier', LinearRegression())]),
            'rf': Pipeline(steps=[('preprocessor', processed_data), ('classifier', RandomForestRegressor())]),
            'svr': Pipeline(steps=[('preprocessor', processed_data), ('classifier', SVR())]),
        }

        # Define the hyperparameters to try for each model
        param_grids = {
            'lr': {
                'classifier__normalize': [True, False],
            },
            'rf': {
                'classifier__n_estimators': [10, 50, 100],
                'classifier__max_depth': [None, 10, 20],
            },
            'svr': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf'],
            },
        }

        # Split the data
        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Train the models and benchmark
        best_model = self.train_models(pipelines, param_grids)

        # Serialize the best model
        self.serialize_model(best_model)
        
        
    