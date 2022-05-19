import pandas as pd
import numpy as np

actors = pd.read_csv('datasets/movie-voice-actors.csv')
movies_count = actors['movie'].value_counts()

scores = {}

total = 0
count = 0

for i, mov in actors.iterrows():
    total += mov['ActorPop']

    if i == 0:
        continue

    if mov['movie'] != actors.at[i - 1, 'movie']:
        total -= mov['ActorPop']

        try:
            avg = total / movies_count[actors.at[i - 1, 'movie']]
            scores[actors.at[i - 1, 'movie']] = avg
        except:
            print(i)

        total = mov['ActorPop']

scores[actors.at[len(actors)-1, 'movie']] = total / movies_count[actors.at[len(actors)-1, 'movie']]

movies_revenue = pd.read_csv('datasets/movies-revenue.csv')
movies_revenue['ActorsAvg'] = np.nan

for i, mov in movies_revenue.iterrows():
    try:
        movies_revenue.iat[i, 9] = scores[mov['title']]
    except:
        movies_revenue.iat[i, 9] = np.nan

movies_revenue.to_csv('datasets/movies-revenue.csv')
