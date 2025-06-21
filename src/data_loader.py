# Load raw data
import pandas as pd
import yaml

def load_data():

    with open('/home/uzokmurod/Desktop/amaliyot/telco-churn/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    raw_data = config['data']['raw_data']
    df = pd.read_csv(raw_data)
    return df