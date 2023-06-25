import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC

# Özellikler
features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "speechiness", "valence"]

# Veri yükleme
dataframe = pd.read_csv('Source/dataset.csv')

#undersampler
dataframe['popularity'] = dataframe['popularity'].apply(lambda x: 1 if x > 64 else 0)

# X ve y ayırma
X = dataframe[features]
y = dataframe['popularity']

# Eğitim ve test veri setlerini oluşturma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline for preprocessing and modeling
pipeline = make_pipeline(StandardScaler(), RandomOverSampler(), RandomForestClassifier())

# Train the model
pipeline.fit(X_train, y_train)

# Predict probabilities and labels
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]  # Probabilities of positive class
y_pred = pipeline.predict(X_test)

# Calculate AUC score
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC Score: {auc_score:.6f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

# Create confusion matrix
# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')  # Set fmt='d' to display decimal integers
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Print classification report
print(classification_report(y_test, y_pred))
