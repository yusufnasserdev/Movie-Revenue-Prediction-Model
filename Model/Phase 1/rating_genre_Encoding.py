import pandas as pd

# MPAA_rating and genre columns pre-processing by get_dummies

movies_revenue = pd.read_csv('datasets/movies-revenue.csv', encoding='unicode_escape')

# generate binary values using get_dummies
movies_revenue = pd.get_dummies(movies_revenue, columns=["MPAA_rating"], prefix=["_is"])
movies_revenue = pd.get_dummies(movies_revenue, columns=["genre"], prefix=["_is"])

movies_revenue.to_csv('datasets/movies-revenue.csv')
