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
dataframe = pd.read_csv('Source/tracks.csv')

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


features = ["acousticness", "danceability", "duration_ms", "energy", "instrumentalness", "liveness",
            "speechiness","loudness", "valence","loudness_instrumentalness_diff" , "energy_instrumentalness_diff", "danceability_energy_diff","popularity"]

data = dataframe[features]


data.to_csv('Source/new_tracks.csv', index=False)

