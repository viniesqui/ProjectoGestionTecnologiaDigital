from train import ModelTrainerRegression

def main():
    trainer = ModelTrainerRegression(
        'Car_Prices_Poland_Kaggle.csv', 
        'price', 
        ['Unnamed: 0', 'mark', 'generation_name', 'city', 'province', 'vol_engine'],
        ['fuel', 'model'], 
        'model.joblib'
    )
    trainer.preprocessor.save_preprocessed_data('preprocessed_data.csv')  # Nuevo csv con datos preprocesados
    trainer.run()

if __name__ == "__main__":
    main()