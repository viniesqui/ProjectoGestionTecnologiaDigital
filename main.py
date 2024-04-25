from train import ModelTrainerRegression
from preprocessing import DataPreprocessor, ReadCsv
    
if __name__ == "__main__":
    
    change_data = ReadCsv('Car_Prices_Poland_Kaggle.csv').preprocess_and_save_data('preprocessed_data.csv', [
        'Unnamed: 0','mark', 'vol_engine', 'generation_name', 'fuel', 'city', 'province'], ['year', 'model'])
   
    nueva_data = ReadCsv('preprocessed_data.csv').read_data()
    
    trainer = ModelTrainerRegression(
        nueva_data,
        'price',
        'model.joblib'
    )
    trainer.run()
# , mark,model,generation_name,year,mileage,vol_engine,fuel,city,province,price