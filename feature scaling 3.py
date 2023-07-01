# -*- coding: utf-8 -*-
"""
Created on Wed May 24 19:54:06 2023

@author: Tahsin
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Özellikler
features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "speechiness", "valence"]

# Veri yükleme
dataframe = pd.read_csv('Source/out.csv')

# Popülerlik sınıflandırması
dataframe['popularity'] = dataframe['popularity'].apply(lambda x: 1 if x > 57 else 0)

# X ve y ayırma
X = dataframe[features]
y = dataframe['popularity']

# Süreyi dakikaya dönüştürme
X['duration_ms'] = X['duration_ms'] / 1000

# Eğitim ve test veri setlerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Farklı feature scaling modellerini deneyelim
scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler()
}

for scaler_name, scaler in scalers.items():
    # Özellikleri ölçeklendirme
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Farklı sınıflandırma modellerini deneyelim
    classifiers = {
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(),
        "Random Forest": RandomForestClassifier()
    }

    print(f"Results using {scaler_name} scaling:")
    for classifier_name, classifier in classifiers.items():
        # Eğitim
        classifier.fit(X_train_scaled, y_train)

        # Tahminler
        y_pred = classifier.predict(X_test_scaled)

        # Performans ölçütleri
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Sonuçları yazdırma
        print(f"Classifier: {classifier_name}\tAccuracy: {accuracy:.6f}\tPrecision: {precision:.6f}\tRecall: {recall:.6f}\tF1-Score: {f1:.6f}")
