import pandas as pd
from dateutil.parser import parse

# Formatting the release_date to match the TMDb date format

movies = pd.read_csv('datasets/movies-revenue-classification.csv')

for i, movie in movies.iterrows():
    # Parse date from a string and return a datetime.datetime
    movie['release_date'] = parse(movie['release_date'])
    # Remove the time from it reducing it to just the date
    movie['release_date'] = movie['release_date'].date()
    # Editing the value at the original dataframe
    movies.at[i, 'release_date'] = movie['release_date']

movies.to_csv('datasets/test.csv', index=False)
