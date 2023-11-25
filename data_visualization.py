# Use plotly for visualizations
# 

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from data_preprocess import preprocess_data

fig_array = []
def visualize_data():
  data, categorical_features, numerical_features = preprocess_data()

  for numerical_feature in numerical_features:

  for categorical_feature in categorical_features:
