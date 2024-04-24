from fastapi import FastAPI, Depends
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from joblib import load
import numpy as np

class Car(BaseModel):
    year: int
    mileage: int
    model: None
    fuel: None

app = FastAPI()

class Model:
    def __init__(self):
        self.model = load('model.joblib')

    def predict(self, data):
        return self.model.predict(data)

def get_model():
    model = Model()
    return model

@app.on_event("startup")
def load_model():
    global model
    model = get_model()

@app.post("/predict")
def predict_price(car: Car, model: Model = Depends(get_model)):
    data = np.array([list(car.dict().values())])
    prediction = model.predict(data)
    return {"predicted_price": prediction[0]}