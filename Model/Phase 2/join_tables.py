import pandas as pd

# reading csv files
movies_base = pd.read_csv('datasets/movies-revenue-classification.csv')
movies_director = pd.read_csv('datasets/movie-director.csv')

# using merge function by setting how='outer'
output = pd.merge(movies_base, movies_director, on='movie_title', how='left')

# displaying result
output.to_csv('datasets/movies-revenue-classification.csv')
