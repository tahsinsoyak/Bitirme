import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, Lasso
import warnings

warnings.filterwarnings("ignore")  # Uyarıları görmezden gelmek için

# Veri setini yükle
dataframe = pd.read_csv('Source/out.csv')

features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness",
            "mode", "speechiness", "tempo", "time_signature", "valence", 'loudness']


data['energy_loudness_ratio'] = data['energy'] / data['loudness']
data['acoustic_energy_ratio'] = data['acousticness'] / data['energy']
data['loudness_instrumentalness_diff'] = abs(data['loudness'] - data['instrumentalness'])
data['energy_instrumentalness_diff'] = abs(data['energy'] - data['instrumentalness'])

# Eğitim ve test veri setlerini ayır
training, test = train_test_split(dataframe, test_size=0.2, random_state=420)

X_train = training[features]
y_train = training['popularity']
X_test = test[features]

# Verileri ölçeklendir
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# DataFrame'i resetle
X_train_scaled = pd.DataFrame(X_train_scaled, columns=features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=features)

correlation_matrix = X_train_scaled.corr().abs()

# Set display options to show all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Print the full correlation matrix
print(correlation_matrix)


# Yeni özelliklerin hesaplanması
X_train_scaled['danceability_energy_diff'] = X_train_scaled['danceability'] - X_train_scaled['energy']
X_test_scaled['danceability_energy_diff'] = X_test_scaled['danceability'] - X_test_scaled['energy']


X_train_scaled['danceability_loudness_diff'] = X_train_scaled['danceability'] - X_train_scaled['loudness']
X_test_scaled['danceability_loudness_diff'] = X_test_scaled['danceability'] - X_test_scaled['loudness']

X_train_scaled['energy_loudness_diff'] = X_train_scaled['energy'] - X_train_scaled['loudness']
X_test_scaled['energy_loudness_diff'] = X_test_scaled['energy'] - X_test_scaled['loudness']


# Seçilecek özelliklerin listesi
selected_features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness",
                     "mode", "speechiness", "tempo", "time_signature", "valence",
                     "danceability_energy_diff", "energy_times_valence", "duration_energy_density",
                     "danceability_loudness_diff", "danceability_loudness_ratio", "energy_loudness_diff",
                     "energy_loudness_ratio"]

# Random Forest Importance ile özellik seçimi
rf = RandomForestRegressor()
rf.fit(X_train_scaled, y_train)
importances = rf.feature_importances_
selected_indices_rf = np.argsort(importances)[::-1][:5]
selected_features_rf = [selected_features[i] for i in selected_indices_rf]

# Boruta ile özellik seçimi
boruta = BorutaPy(estimator=RandomForestRegressor(), n_estimators='auto', max_iter=25)
boruta.fit(X_train_scaled, y_train)
selected_indices_boruta = boruta.support_
selected_features_boruta = [selected_features[i] for i in selected_indices_boruta]

# RFE ile özellik seçimi
model = LinearRegression()
rfe = RFE(estimator=model, n_features_to_select=5)
rfe.fit(X_train_scaled, y_train)
selected_indices_rfe = rfe.support_
selected_features_rfe = [selected_features[i] for i in selected_indices_rfe]

# Lasso Regression ile özellik seçimi
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_scaled, y_train)
selected_indices_lasso = np.nonzero(lasso.coef_)[0]
selected_features_lasso = [selected_features[i] for i in selected_indices_lasso]

# SelectKBest ile özellik seçimi
kbest = SelectKBest(score_func=f_regression, k=5)
kbest.fit(X_train_scaled, y_train)
selected_indices_kbest = kbest.get_support(indices=True)
selected_features_kbest = [selected_features[i] for i in selected_indices_kbest]

# Seçilen özellikleri yazdır
print("Selected Features using Random Forest Importance:")
print(selected_features_rf)
print("\nSelected Features using Boruta:")
print(selected_features_boruta)
print("\nSelected Features using Recursive Feature Elimination (RFE):")
print(selected_features_rfe)
print("\nSelected Features using Lasso Regression:")
print(selected_features_lasso)
print("\nSelected Features using SelectKBest:")
print(selected_features_kbest)
