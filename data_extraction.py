import pandas as pd

def load_data():
  data = pd.read_csv(Weather_Data.csv)
  print(data.head())
  return data

load_data()
