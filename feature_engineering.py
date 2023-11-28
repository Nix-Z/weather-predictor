import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_visualization import visualize_data

def engineer_features():
    data = visualize_data()

    # Balance the Data
    new_data = data.drop(columns=['RainToday'])
    scaler = StandardScaler()
    scaler.fit(new_data)
    scaled_features = scaler.transform(new_data)
    scaled_data = pd.DataFrame(scaled_features,columns=new_data.columns[:])
    selected_columns = data[['RainToday']]
    scaled_data[['RainToday']] = selected_columns.copy()

    # Convert categorical feature values to binary values
    scaled_data.replace({'RainToday': {'No':0, 'Yes':1}}, inplace=True)
    print(scaled_data.head())

    data_balanced.to_csv('weather_predictor_cleansed_data.csv', index=False)

    return scaled_data

engineer_features()
