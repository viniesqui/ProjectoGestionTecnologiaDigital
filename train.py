from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from joblib import dump, load
from preprocessing import DataPreprocessor
from sklearn.model_selection import GridSearchCV

class ModelTrainerRegression:
    class ModelTrainerRegression:
        def __init__(self, data_filepath, target_column, columns_to_remove, columns_to_convert, model_filepath='model.joblib'):
            self.data_filepath = data_filepath
            self.target_column = target_column
            self.model_filepath = model_filepath
            self.columns_to_remove = columns_to_remove
            self.columns_to_convert = columns_to_convert
            self.preprocessor = DataPreprocessor(data_filepath, columns_to_remove=self.columns_to_remove, columns_to_convert=self.columns_to_convert)
    
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
        df = self.preprocessor.df

        X = df.drop(self.target_column, axis=1)
        y = df[self.target_column]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        self.X_train = self.preprocessor.preprocess_data(self.X_train)
        self.X_test = self.preprocessor.preprocess_data(self.X_test)

        pipelines = {
            'lr': Pipeline(steps=[('classifier', LinearRegression())]),
            'rf': Pipeline(steps=[('classifier', RandomForestRegressor())]),
            'svr': Pipeline(steps=[('classifier', SVR())]),
        }

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

        best_model = self.train_models(pipelines, param_grids)

        self.serialize_model(best_model)