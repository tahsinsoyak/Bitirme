# -*- coding: utf-8 -*-
"""
Created on Thu May 25 11:55:37 2023

@author: Tahsin
"""


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Özellikler
features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "speechiness", "valence"]

# Veri yükleme
dataframe = pd.read_csv('Source/tracks.csv')


sns.distplot(dataframe['popularity']).set_title('Popularity Distribution')



# X ve y ayırma
X = dataframe[features]
y = dataframe['popularity']

#kfold eklenecek

# Eğitim ve test veri setlerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train['danceability_energy_diff'] = X_train['danceability'] - X_train['energy']
X_test['danceability_energy_diff'] = X_test['danceability'] - X_test['energy']


X_train['energy_times_valence'] = X_train['energy'] * X_train['valence']
X_test['energy_times_valence'] = X_test['energy'] * X_test['valence']


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

