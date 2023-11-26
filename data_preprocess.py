from data_analysis import analyze_data

def preprocess_data():
  data, categorical_features, numerical_features = analyze_data()
  data.drop(['Date', 'WindGustDir', 'WindDir9am', 'WindDir3pm'], axis=1, inplace=True)
  print(data.head())
  categorical_features = data.select_dtypes("object").columns
  numerical_features = data.select_dtypes("number").columns
  return data, categorical_features, numerical_features

preprocess_data()
