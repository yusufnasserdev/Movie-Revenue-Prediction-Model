import pandas as pd
import numpy as np
import keys

from tmdbv3api import Person
from tmdbv3api import TMDb

# Creating a base class instance from the api library
tmdb = TMDb()
tmdb.api_key = keys.tmdb_key
tmdb.language = 'en'
tmdb.debug = True

# Creating a Movie instance to search by the movie details
person = Person()

movies = pd.read_csv('datasets/movies-revenue.csv')

movies['DirectorPop'] = np.nan

for i, mov in movies.iterrows():
    search = person.search(mov['director'])  # Search by the movie title
    for res in search:
        movies.iat[i, 7] = res['popularity']
        break

movies.to_csv('datasets/movies-revenue.csv')
