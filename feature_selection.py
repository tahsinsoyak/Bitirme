# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 12:37:19 2023

@author: Tahsin
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Lasso
import warnings

warnings.filterwarnings("ignore")

# Load the dataset
dataframe = pd.read_csv('Source/out.csv')

features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness",
            "mode", "speechiness", "tempo", "time_signature", "valence", 'loudness',"popularity"]

# Perform feature engineering
dataframe['energy_loudness_ratio'] = dataframe['energy'] / dataframe['loudness']
dataframe['acoustic_energy_ratio'] = dataframe['acousticness'] / dataframe['energy']
dataframe['loudness_instrumentalness_diff'] = abs(dataframe['loudness'] - dataframe['instrumentalness'])
dataframe['energy_instrumentalness_diff'] = abs(dataframe['energy'] - dataframe['instrumentalness'])
dataframe['danceability_energy_diff'] = abs(dataframe['danceability'] - dataframe['energy'])
dataframe['duration_energy_density'] = dataframe['duration_ms'] / dataframe['energy']


# Use the updated dataframe for further analysis
dataframe = dataframe[features + ['energy_loudness_ratio', 'acoustic_energy_ratio', 'loudness_instrumentalness_diff', 'energy_instrumentalness_diff',"danceability_energy_diff",'duration_energy_density']]

dataframe = dataframe.replace([np.inf, -np.inf], np.nan).dropna()

data = dataframe

# Split the data into features and target
X = data.drop('popularity', axis=1)  # Replace 'target_column_name' with the actual target column name
y = data['popularity']

# let's initialize a RF model 
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

# let's initialize Boruta
feat_selector = BorutaPy(
    verbose=2,
    estimator=model,
    n_estimators='auto',
    max_iter=10  # number of iterations to perform
)

# train Boruta
# N.B.: X and y must be numpy arrays
feat_selector.fit(np.array(X), np.array(y))

# print support and ranking for each feature
print("\n------Support and Ranking for each feature------")
for i in range(len(feat_selector.support_)):
    if feat_selector.support_[i]:
        print("Passes the test: ", X.columns[i],
              " - Ranking: ", feat_selector.ranking_[i])
    else:
        print("Doesn't pass the test: ",
              X.columns[i], " - Ranking: ", feat_selector.ranking_[i])