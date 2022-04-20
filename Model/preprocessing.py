import numpy as np
import pandas as pd

"""
# Director column pre-processing by get_dummies

movies_revenue = pd.read_csv('datasets/movies-revenue.csv', encoding='unicode_escape')

# generate binary values using get_dummies
movies_revenue = pd.get_dummies(movies_revenue, columns=["director"], prefix=["_is"])
movies_revenue.to_csv('datasets/movies-revenue.csv')
"""
# Animation or Not column

movies_revenue = pd.read_csv('datasets/movies-revenue.csv', encoding='unicode_escape')
movies_revenue['CharactersCount'] = np.nan
movies_revenue['IsAnimation'] = np.nan

voice_actors = pd.read_csv('datasets/movie-voice-actors.csv', encoding='unicode_escape')
voice_actors.columns = ['character', 'voice-actor', 'title']

actors_freq = voice_actors['title'].value_counts()

for i, mov in movies_revenue.iterrows():
    try:
        movies_revenue.iat[i, 7] = actors_freq[mov['title']]
        movies_revenue.iat[i, 8] = 1
    except:
        movies_revenue.iat[i, 7] = 0
        movies_revenue.iat[i, 8] = 0

movies_revenue.to_csv('datasets/movies-revenue.csv')
