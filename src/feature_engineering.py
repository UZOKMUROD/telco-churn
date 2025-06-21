# Feature engineering utilities
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from data_loader import load_data

class DataProcess(BaseEstimator, TransformerMixin):

    def fit(self):
        return self
    
    def transform(self, df):
        
        df.drop(columns=['customerID', 'gender'], inplace=True, axis=1)
        df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex = True)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
        df["TotalCharges"] = df["TotalCharges"].ffill()
        df = df.drop_duplicate()
            
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

def pipeline(self):
    pipeline = Pipeline([
        ("Data cleaner", DataProcess()),
        ('Encoder', OneHotEncoder())
    ])