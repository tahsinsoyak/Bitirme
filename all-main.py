# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:44:23 2023

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

# Özellikler
features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "speechiness", "valence"]

# Veri setini yükleme
dataframe = pd.read_csv('Source/tracks.csv')

# Boş değerleri içeren satırları silme
dataframe.dropna(inplace=True)


dataframe.describe()

class_counts = dataframe['popularity'].value_counts()
with pd.option_context('display.max_rows', None):
    print(class_counts)

# Veri setini sınıflandırma için hazırlama
popularity_threshold = dataframe['popularity'].median()
dataframe['popularity'] = dataframe['popularity'].apply(lambda x: 1 if x > 50 else 0)

# X ve y ayırma
X = dataframe[features]
y = dataframe['popularity']

# Eğitim ve test veri setlerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sınıflandırma modelleri
models = [
    RandomForestClassifier(),
    DecisionTreeClassifier(),
    KNeighborsClassifier(),
    LogisticRegression(),
    XGBClassifier(),
    LinearSVC()
]

# Performans ölçümlerini saklamak için bir liste oluşturma
performance_scores = []

# Modelleri eğitme ve performans ölçümü
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    performance_scores.append((model.__class__.__name__, accuracy))

# Performans skorlarına göre sıralama
performance_scores.sort(key=lambda x: x[1], reverse=True)

# Sıralanmış performans skorlarını yazdırma
for model_name, accuracy in performance_scores:
    print(f"{model_name}\t{accuracy:.6f}")
