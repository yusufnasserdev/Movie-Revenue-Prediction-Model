import numpy as np
import pandas as pd
import sklearn as sk

# reading csv files
movies_base = pd.read_csv('datasets/movies-revenue.csv')
movies_director = pd.read_csv('datasets/movie-director.csv')

# using merge function by setting how='outer'
output4 = pd.merge(movies_base, movies_director, on='title', how='left')

# displaying result
output4.to_csv('datasets/movies-revenue.csv')
