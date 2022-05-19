from tmdbv3api import TMDb
from tmdbv3api import Movie
import pandas as pd

# Using the TMDb to fill out the missing director from the original dataset

# https://github.com/AnthonyBloomer/tmdbv3api
# https://developers.themoviedb.org/3/getting-started/introduction

# Creating a base class instance from the api library
tmdb = TMDb()
tmdb.api_key = 'b5ebbeb68bbdb72e376724a36cf7dc0f'
tmdb.language = 'en'
tmdb.debug = True

# Creating a Movie instance to search by the movie details
movie = Movie()

movies = pd.read_csv('datasets/movies-revenue.csv')

for i, mov in movies.iterrows():
    if type(mov['director']) == float:  # Float is the default datatype for an empty cell
        search = movie.search(mov['title'])  # Search by the movie title
        for res in search:
            try:
                # Confirming the search results by the release date
                if res['release_date'][:7] == mov['release_date'][:7]:
                    # Extracting the director from the movie credits
                    for z in movie.credits(res.id)['crew']:
                        if z['job'] == 'Director':
                            # Editing the value at the original dataframe
                            movies.iat[i, 6] = z['name']
                            break
            except BaseException as error:
                print('An exception occurred: {}'.format(error) + " " + mov['title'])

movies.to_csv('datasets/movies-revenue.csv', index=False)
