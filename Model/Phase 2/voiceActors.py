import pandas as pd
import numpy as np

from tmdbv3api import Person
from tmdbv3api import TMDb

actors = pd.read_csv('datasets/movie-voice-actors.csv')
actors.sort_values(['movie_title'], axis=0, inplace=True)

actors.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
actors.to_csv('datasets/movie-voice-actors.csv')

# Creating a base class instance from the api library
tmdb = TMDb()
tmdb.api_key = 'b5ebbeb68bbdb72e376724a36cf7dc0f'
tmdb.language = 'en'
tmdb.debug = True

# Creating a Movie instance to search by the movie details
person = Person()

actors = pd.read_csv('datasets/movie-voice-actors.csv')

actors['ActorPop'] = np.nan

for i, mov in actors.iterrows():
    if str(mov['voice-actor']).find(';') != -1:
        l = mov['voice-actor'].split(';')
        avg = 0.0
        count = 0
        for j in l:
            search = person.search(j)  # Search by the movie title
            try:
                avg += search[0]['popularity']
                count += 1
            except:
                print("Not Found ", j)
        actors.at[i, 'ActorPop'] = avg / count
    else:
        search = person.search(mov['voice-actor'])  # Search by the actor name
        try:
            actors.at[i, 'ActorPop'] = search[0]['popularity']
        except:
            print("Not Found", i)

actors.to_csv('datasets/movie-voice-actors.csv')

actors = pd.read_csv('datasets/movie-voice-actors.csv')
movies_count = actors['movie_title'].value_counts()

scores = {}

total = 0
count = 0

for i, mov in actors.iterrows():
    total += mov['ActorPop']

    if i == 0:
        continue

    if mov['movie_title'] != actors.at[i - 1, 'movie_title']:
        total -= mov['ActorPop']

        try:
            avg = total / movies_count[actors.at[i - 1, 'movie_title']]
            scores[actors.at[i - 1, 'movie_title']] = avg
        except:
            print(i)

        total = mov['ActorPop']

scores[actors.at[len(actors)-1, 'movie_title']] = total / movies_count[actors.at[len(actors)-1, 'movie_title']]

movies_revenue = pd.read_csv('datasets/movies-revenue.csv')
movies_revenue['ActorsAvg'] = np.nan

for i, mov in movies_revenue.iterrows():
    try:
        movies_revenue.at[i, 'ActorsAvg'] = scores[mov['movie_title']]
    except:
        movies_revenue.at[i, 'ActorsAvg'] = np.nan

movies_revenue.to_csv('datasets/movies-revenues.csv')

