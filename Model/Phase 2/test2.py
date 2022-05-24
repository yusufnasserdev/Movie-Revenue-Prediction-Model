import warnings

import numpy as np
import pandas as pd

from re import sub
from decimal import Decimal
from dateutil.parser import parse
import seaborn as sns

import matplotlib.pyplot as plt
import pickle
import time

from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures

warnings.filterwarnings("ignore")

# Reading csv files
movies = pd.read_csv('test_sets/movies-revenue-classification.csv')
movies_director = pd.read_csv('test_sets/movie-director.csv')

# Merging datasets
movies = pd.merge(movies, movies_director, on='movie_title', how='left')

# Preprocessing release_date
for i, movie in movies.iterrows():
    # Parse date from a string and return a datetime.datetime
    movie['release_date'] = parse(movie['release_date'])
    # Remove the time from it reducing it to just the date
    movie['release_date'] = movie['release_date'].date()
    # Editing the value at the original dataframe
    x = str(movie['release_date'])
    # Splitting the date
    date_split = x.split('-')
    # Calculating the months and days
    calc = (((float(date_split[1]) - 1) * 30) + float(date_split[2])) / 365
    # Adding calc to the years
    date_final = float(date_split[0]) + calc
    # Replacing the date by its decimal value
    movies.at[i, 'release_date'] = date_final

# Preprocessing Director and Voice Actors
movies['DirectorPop'] = np.nan
movies['ActorsAvg'] = np.nan

dirs = pd.read_csv('datasets/movies-revenue-classification.csv')

for i, mov in movies.iterrows():
    for j, old in dirs.iterrows():
        if mov['movie_title'] == old['movie_title']:
            movies.at[i, 'director_pop'] = old['director_pop']
            movies.at[i, 'ActorsAvg'] = old['ActorsAvg']
            break

# Filling the missing data with the mean value
director_mean_value = movies['director_pop'].mean()
movies['director_pop'].fillna(value=director_mean_value, inplace=True)

Actors_mean_value = movies['ActorsAvg'].mean()
movies['ActorsAvg'].fillna(value=Actors_mean_value, inplace=True)

# Preprocessing Voice Actors
movies['IsAnimation'] = np.nan

voice_actors = pd.read_csv('test_sets/movie-voice-actors.csv', encoding='unicode_escape')
voice_actors.columns = ['character', 'voice-actor', 'movie_title']

actors_freq = voice_actors['movie_title'].value_counts()

for i, mov in movies.iterrows():
    try:
        movies.at[i, 'IsAnimation'] = 1
    except:
        movies.at[i, 'IsAnimation'] = 0

# Preprocessing genre and MPAA_rating
movies = pd.get_dummies(movies, columns=["MPAA_rating"], prefix=["_is"])
movies = pd.get_dummies(movies, columns=["genre"], prefix=["_is"])

# Dividing Data
X = movies  # Features

Y = X['MovieSuccessLevel']  # Label

# Preparing Data
dirs = pd.read_csv('datasets/movies-revenue-classification.csv')
X = X.reindex(labels=dirs.columns, axis=1)

X.drop(['movie_title', 'director', 'ActorsAvg'], axis=1, inplace=True)
Y = np.where(Y == "S", 5, np.where(Y == "A", 4, np.where(Y == "B", 3, np.where(Y == "C", 2, 1))))
X['MovieSuccessLevel'] = Y

# Dropping the label from the features dataframe
X.drop(['MovieSuccessLevel'], axis=1, inplace=True)

# Feature Scaling
scaler = MinMaxScaler()
dirs.drop(['movie_title', 'director', 'ActorsAvg', 'MovieSuccessLevel'], axis=1, inplace=True)
scaler.fit(dirs)
X = scaler.transform(X)

pickled_model = pickle.load(open('DT.pkl', 'rb'))

start_test = time.time()
prediction = pickled_model.predict(X)
end_test = time.time()

print("Accuracy Decision Tree:", metrics.accuracy_score(Y, prediction))
print("Actual time for Testing", end_test - start_test)
