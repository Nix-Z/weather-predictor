import pandas as pd
from data_extraction import analyze_data

def process_data():
  data, categorical_features, numerical_features = analyze_data()
  data['Date'] = pd.to_datetime(data['Date'], format="%d-%m-%y")
  
