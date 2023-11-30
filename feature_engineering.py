import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from datavisualization import visualize_data

def engineer_features():
    data = visualize_data()

    # Remove Outliers
    outliers = ['Temp9am', 'Temp3pm', 'MaxTemp', 'Evaporation', 'WindGustSpeed', 'WindSpeed9am',
                'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm']
    for feature in outliers:
        percentile_25 = data[feature].quantile(0.25)
        percentile_75 = data[feature].quantile(0.75)
        iqr = percentile_75 - percentile_25
        upper_limit = percentile_75 + 1.5 * iqr
        lower_limit = percentile_25 - 1.5 * iqr
        data[feature] = np.where(data[feature] > upper_limit, upper_limit,
                                 np.where(data[feature] < lower_limit, lower_limit, data[feature]))
        data[feature] = data[feature].astype(int)

    # Balance the Data
    new_data = data.drop(columns=['RainToday'])
    scaler = StandardScaler()
    scaler.fit(new_data)
    scaled_features = scaler.transform(new_data)
    scaled_data = pd.DataFrame(scaled_features,columns=new_data.columns[:])
    selected_columns = data[['RainToday']]
    scaled_data[['RainToday']] = selected_columns.copy()
    data = scaled_data

    # Convert categorical feature values to binary values
    data.replace({'RainToday': {'No':0, 'Yes':1}}, inplace=True)
    print(data.head())

    # Balance data --> RainToday
    data_majority = data[data['RainToday']==0]
    data_minority = data[data['RainToday']==1]
    data_minority_upsampled = resample(data_minority, replace=True, n_samples=2422, random_state=1)
    data_upsampled = pd.concat([data_majority, data_minority_upsampled], axis=0)

    print(data_upsampled['RainToday'].groupby(data_upsampled['RainToday']).count())
    print(data_upsampled.head())

    data_upsampled.to_csv('weather_predictor_cleansed_data.csv', index=False)
    
    return data_upsampled

engineer_features()
