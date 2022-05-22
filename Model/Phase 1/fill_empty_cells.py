import pandas as pd

movies = pd.read_csv('datasets/movies-revenue.csv')

# Finding the mean of the column having NaN
director_mean_value = movies['DirectorPop'].mean()
Actors_mean_value = movies['ActorsAvg'].mean()
CharactersCount_mean_value = movies['CharactersCount'].mean()

movies['DirectorPop'].fillna(value=director_mean_value, inplace=True)
movies['ActorsAvg'].fillna(value=Actors_mean_value, inplace=True)
movies['CharactersCount'].fillna(value=CharactersCount_mean_value, inplace=True)

movies.to_csv('datasets/movies-revenue.csv', index=False)
