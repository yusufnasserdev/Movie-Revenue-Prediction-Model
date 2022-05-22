import pandas as pd

movies = pd.read_csv('datasets/movies-revenue-classification.csv')

# Finding the mean of the column having NaN
director_mean_value = movies['director_pop'].mean()
Actors_mean_value = movies['ActorsAvg'].mean()

movies['director_pop'].fillna(value=director_mean_value, inplace=True)
movies['ActorsAvg'].fillna(value=Actors_mean_value, inplace=True)

movies.to_csv('datasets/movies-revenue.csv', index=False)
