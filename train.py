# Main training script
from src.model import Train_models
from src.feature_engineering import DataProcess
from src.data_loader import load_data
import pandas as pd
data_process = DataProcess()

def full_run():
    df = load_data()

    X, y = data_process.transform(df)
     
    train_models = Train_models(X=X, y=y)
    
    test_results = train_models.run()
    best_model, _ = train_models.best_model()
    
    return best_model, test_results



if __name__ == '__main__':
    best_model, test_results = full_run()
    
    print(test_results)
    
    print(f'\nAfter finetune Best model name is {best_model}')