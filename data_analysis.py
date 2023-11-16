import pandas as pd
from data_extraction import load_data

def analyze_data():
  data = load_data()
  print(data.describe())
  print(data.info())
  print(f"Data Features: {data.columns}") # Lists data features
  check_values = []
  for column in data.columns:
    #print(column, data[column].isnull().sum()) # Number of null values per feature
    #print(column, data[column].duplicated().sum())  # Number of duplicated values per feature
    #print(column, data[column].nunique()) # Number of unique values per feature

    data_types = data[column].dtypes # dtype of feature
    sum_null = data[column].isnull().sum()
    sum_dup = data[column].duplicated().sum()
    unique_values = data[column].nunique()
    check_values.append([column, data_types, sum_null, sum_dup, unique_values,])
  check_values = pd.DataFrame(1)
  check_values.columns = ['Feature', 'Data Type', '# of Null Values', '# of Duplicate Values', '# of Unique Values']
  print(check_values)
  categorical_features = data.select_dtypes("object").columns
  numerical_features = data.select_dtypes("number").columns
  return data, categorical_features, numerical_features
  
analyze_data()
