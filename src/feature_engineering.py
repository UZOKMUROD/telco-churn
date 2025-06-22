# Feature engineering utilities
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.data_loader import load_data

df = load_data()
class DataProcess(BaseEstimator, TransformerMixin):

    def fit(self):
        return self
    
    def transform(self, data):
        print(data)
        df = data
        df.drop(columns=['customerID', 'gender'], inplace=True, axis=1)
        df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex = True)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
        df["TotalCharges"] = df["TotalCharges"].ffill()
        df = df.drop_duplicates()
            
        object_columns = df.select_dtypes(include = 'object').columns
        
        for col in object_columns:
            unique_val = df[col].unique()
            values = []
            i = 2
            for uc in unique_val:
                 
                if uc == 'Yes':
                    values.append(1)
                elif uc == "No":
                    values.append(0)
                else:
                    values.append(i)
                    i+=1
        
            df[col] = df[col].replace(unique_val, values)
        
        y = df.Churn
        X = df.drop('Churn', axis = 1)
        
        return X, y
    
if __name__ == '__main__':
    data_process = DataProcess()
    data_process.transform(df)
    