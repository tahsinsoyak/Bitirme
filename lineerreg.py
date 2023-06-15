import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Özellikler
features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "speechiness", "valence"]

# Veri yükleme
dataframe = pd.read_csv('Source/out.csv')

# X ve y ayırma
X = dataframe[features]
y = dataframe['popularity']

# Eğitim ve test veri setlerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Descent için parametrelerin başlangıç değerleri
theta = np.zeros(X_train.shape[1] + 1)

# Gradient Descent hiperparametreleri
learning_rate = 0.01
num_iterations = 1000

# Fonksiyonlar

def add_intercept(X):
    """Özellik matrisine bir sütun ekler (bias terimi için)"""
    intercept = np.ones((X.shape[0], 1))
    return np.concatenate((intercept, X), axis=1)

def predict(X, theta):
    """Tahminleri yapar"""
    X_intercept = add_intercept(X)
    return np.dot(X_intercept, theta)

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    """Gradient Descent algoritması"""
    m = len(y)
    history = []  # Her iterasyon sonrası maliyeti kaydetmek için
    
    for iteration in range(num_iterations):
        # Tahminleri yap
        y_pred = predict(X, theta)
        
        # Gradyanı hesapla
        gradient = np.dot(X.T, (y_pred - y)) / m
        
        # Parametreleri güncelle
        theta -= learning_rate * gradient
        
        # Her iterasyon sonrası maliyeti kaydet
        cost = mean_squared_error(y, y_pred)
        history.append(cost)
        
    return theta, history

# Özellik matrisine intercept sütunu ekleme
X_train_intercept = add_intercept(X_train)

# Gradient Descent ile parametreleri güncelleme
theta, cost_history = gradient_descent(X_train_intercept, y_train, theta, learning_rate, num_iterations)

# Test veri seti üzerinde tahmin yapma
X_test_intercept = add_intercept(X_test)
y_pred = predict(X_test_intercept, theta)

# Tahminleri sınıflara dönüştürme
y_pred_class = np.where(y_pred > 0.5, 1, 0)

# Performans ölçütleri
accuracy = accuracy_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)

# Sonuçları yazdırma
print(f"Accuracy: {accuracy:.6f}\tPrecision: {precision:.6f}\tRecall: {recall:.6f}\tF1-Score: {f1:.6f}")

# Maliyetin iterasyonlar boyunca nasıl değiştiğini görselleştirme
plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Gradient Descent Cost')
plt.show()
