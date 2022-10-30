import pandas as pd
import keys

from tmdbv3api import TMDb
from tmdbv3api import Movie
from dateutil.parser import parse

# Using the TMDb to fill out the missing director from the original dataset

# https://github.com/AnthonyBloomer/tmdbv3api
# https://developers.themoviedb.org/3/getting-started/introduction

# Creating a base class instance from the api library
tmdb = TMDb()
tmdb.api_key = keys.tmdb_key
tmdb.language = 'en'
tmdb.debug = True

# Creating a Movie instance to search by the movie details
movie = Movie()

movies = pd.read_csv('datasets/movies-revenue-classification.csv')

for i, mov in movies.iterrows():
    if type(mov['director']) == float:  # Float is the default datatype for an empty cell    search = movie.search(mov['movie_title'])  # Search by the movie title
        search = movie.search(mov['movie_title'])  # Search by the movie title
        for res in search:
            try:
                x = res['release_date']
                x = parse(x)
                x = x.date()

                y = mov['release_date']
                y = parse(y)
                y = y.date()

                # Confirming the search results by the release date
                if str(x)[:7] == str(y)[:7]:
                    # Extracting the director from the movie credits
                    for z in movie.credits(res.id)['crew']:
                        if z['job'] == 'Director':
                            # Editing the value at the original dataframe
                            movies.at[i, 'director'] = z['name']
                            movies.at[i, 'director_pop'] = z['popularity']
                            break
            except BaseException as error:
                print('An exception occurred: {}'.format(error) + " " + mov['movie_title'])

print(movies)
movies.to_csv('datasets/movies-revenue.csv', index=False)
