from re import sub
from decimal import Decimal
import pandas as pd
movies = pd.read_csv('datasets/movies-revenue.csv')
for i, mov in movies.iterrows():
    movies.at[i, 'revenue'] = Decimal(sub(r'[^\d.]', '', mov['revenue']))

movies.to_csv('datasets/movies-revenue.csv')