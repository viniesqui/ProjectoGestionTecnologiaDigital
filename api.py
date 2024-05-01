from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from train import ModelTrainerRegression

app = FastAPI()

class Item(BaseModel):
    model: str
    year: int
    mileage: int
    fuel: str

@app.post("/add_data/")
async def add_data(item: Item):
    data = item.dict()
    df = pd.read_csv('preprocessed_data.csv')
    df = df.append(data, ignore_index=True)
    df.to_csv('preprocessed_data.csv', index=False)
    if len(df) % 100 == 0:
        retrain_model(df)
    return {"message": "Data added and model retrained if necessary"}

def retrain_model(df):
    target_column = 'price'  
    model_filepath = 'model.joblib' 
    trainer = ModelTrainerRegression(df, target_column, model_filepath)
    trainer.run()