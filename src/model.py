# Model training
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier



class Train_models:
    def __init__(self, X, y, test_size=0.2, cv=5, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.cv = cv
        self.random_state = random_state
        self.result = {}
        self.trained_models = {}
        
        self.models = {
            'Xgboost': XGBClassifier(),
            'LGBMC': LGBMClassifier(),
            'DecisionTreeClassifier':DecisionTreeClassifier(random_state=random_state),
            'GradientBoostingClassifier': GradientBoostingClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
        }
        
    def run(self, scale = True):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_state)
        
        for name, model in self.models.items():
            
            if scale:
                pipeline = make_pipeline(StandardScaler(), model)
            else:
                pipeline = model
                
            pipeline.fit(X_train, y_train)
            self.trained_models[name] = pipeline
            
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
          
            
            self.result[name] = {
                'accuracy_score': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc_score':roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
            }
        
        self.result_df = pd.DataFrame(self.result).T.sort_values('roc_auc_score', ascending=False)
        
        return self.result_df
    
    def get_model(self, name):
        
        return self.trained_models.get(name)
    
    def best_model(self, metrics = 'roc_auc_score'):
        if not hasattr(self, 'result_df'):
            raise ValueError('You call .run() function first!!!')
        best_model_name = self.result_df[metrics].idxmax()
        return best_model_name, self.trained_models[best_model_name]
