import pandas as pd
from data_extraction import load_data

def analyze_data():
  data = load_data()
  print(data.describe())
  print(data.info())
  print(f"Data Features: {data.columns}") # Lists data features
  print(data.isnull().sum()) # Number of null values per feature
  print(data.duplicated().sum()) # Number of duplicated values per feature
  for column in data.columns:
    print(column, data[column].nunique()) # Number of unique values per feature
  categorical_features = data.select_dtypes("object").columns
  print(f"Categorical Features: {categorical_features}")
  numerical_features = data.select_dtypes("number").columns
  print(f"Numerical Features: {numerical_features}")
  return data, categorical_features, numerical_features
  
analyze_data()
