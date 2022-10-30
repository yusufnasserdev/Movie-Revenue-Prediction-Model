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
        actors.iat[i, 3] = avg / count
    else:
        search = person.search(mov['voice-actor'])  # Search by the movie title
        try:
            actors.iat[i, 3] = search[0]['popularity']
        except:
            print("Not Found", i)

actors.to_csv('datasets/movie-voice-actors.csv')
