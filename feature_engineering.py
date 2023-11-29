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

    # Balance data --> RainToday
    data = scaled_data
    
    data_majority = data[data['RainToday']==0]
    data_minority = data[data['RainToday']==1]
    data_minority_upsampled = resample(data_minority, replace=True, n_samples=2422, random_state=1)
    data_upsampled = pd.concat([data_majority, data_minority_upsampled], axis=0)

    print(data_upsampled['RainToday'].groupby(data_upsampled['RainToday']).count())
    print(data_upsampled.head())

    data_upsampled.to_csv('weather_predictor_cleansed_data.csv', index=False)
    
    return data_upsampled

engineer_features()
