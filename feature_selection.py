from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv('Source/out.csv')

features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "key", "liveness",
            "mode", "speechiness", "tempo", "time_signature", "valence"]

training = dataframe.sample(frac=0.8, random_state=420)
X_train = training[features]
y_train = training['popularity']
X_test = dataframe.drop(training.index)[features]

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=420)

"""

#BORUTA FEATURE SELECTİON

# let's initialize a RF model
model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

# let's initialize Boruta
feat_selector = BorutaPy(
    verbose=2,
    estimator=model,
    n_estimators='auto',
    max_iter=100 # number of iterations to perform
)

# train Boruta
# N.B.: X_train and y_train must be numpy arrays
feat_selector.fit(np.array(X_train), np.array(y_train))

# print support and ranking for each feature
print("\n------Support and Ranking for each feature------")
for i in range(len(feat_selector.support_)):
    if feat_selector.support_[i]:
        print("Passes the test: ", X_train.columns[i],
              " - Ranking: ", feat_selector.ranking_[i])
    else:
        print("Doesn't pass the test: ",
              X_train.columns[i], " - Ranking: ", feat_selector.ranking_[i])

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
import seaborn as sns

# Özellikler
features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "speechiness", "valence"]

sns.distplot(dataframe['popularity']).set_title('Popularity Distribution')
# Veri yükleme
dataframe = pd.read_csv('Source/dataset.csv')


popularity_threshold = dataframe['popularity'].median()
dataframe['popularity'] = dataframe['popularity'].apply(lambda x: 1 if x > 72 else 0)



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