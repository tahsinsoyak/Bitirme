# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 12:32:16 2023

@author: Tahsin
"""

from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

dataframe = pd.read_csv('Source/dataset.csv')

features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness",
            "mode", "speechiness", "tempo", "time_signature", "valence"]

training = dataframe.sample(frac=0.8, random_state=420)
X_train = training[features]
y_train = training['popularity']
X_test = dataframe.drop(training.index)[features]


z_scores = stats.zscore(X_train)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
X_train = X_train[filtered_entries]
y_train = y_train[filtered_entries]

# Calculate additional features
X_train.loc[:, 'danceability_energy_diff'] = X_train['danceability'] - X_train['energy']
X_test.loc[:, 'danceability_energy_diff'] = X_test['danceability'] - X_test['energy']

X_train.loc[:, 'energy_times_valence'] = X_train['energy'] * X_train['valence']
X_test.loc[:, 'energy_times_valence'] = X_test['energy'] * X_test['valence']

X_train.loc[:, 'duration_energy_density'] = X_train['duration_ms'] / X_train['energy']
X_test.loc[:, 'duration_energy_density'] = X_test['duration_ms'] / X_test['energy']

# Perform feature scaling
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=420)



#BORUTA FEATURE SELECTİON

# let's initialize a RF model
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

# let's initialize Boruta
feat_selector = BorutaPy(
    verbose=2,
    estimator=model,
    n_estimators='auto',
    max_iter=10 # number of iterations to perform
)

# train Boruta
# N.B.: X_train and y_train must be numpy arrays
feat_selector.fit(np.array(X_train), np.array(y_train))

# Modify the features list to include additional features
features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness",
            "mode", "speechiness", "tempo", "time_signature", "valence", "danceability_energy_diff",
            "energy_times_valence", "duration_energy_density","tempo_variation","harmonic_complexity", "loudness_dynamics","speechiness_variation","key_changes"]

# Print support and ranking for each feature
print("\n------Support and Ranking for each feature------")
for i in range(len(feat_selector.support_)):
    if feat_selector.support_[i]:
        print("Passes the test: ", features[i],
              " - Ranking: ", feat_selector.ranking_[i])
    else:
        print("Doesn't pass the test: ",
              features[i], " - Ranking: ", feat_selector.ranking_[i])


