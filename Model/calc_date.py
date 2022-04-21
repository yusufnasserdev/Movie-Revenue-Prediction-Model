import pandas as pd

movies = pd.read_csv('datasets/movies-revenue.csv')

for i, mov in movies.iterrows():
    x = str(mov['release_date'])
    l = x.split('/')
    calc = (((float(l[0]) - 1) * 30) + float(l[1])) / 365
    l[2] = float(l[2]) + calc
    movies.iat[i, 1] = l[2]

movies.to_csv('datasets/movies-revenue.csv')