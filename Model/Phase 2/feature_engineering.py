import numpy as np
import pandas as pd

# Animation or Not column

movies_revenue = pd.read_csv('datasets/movies-revenue-classification.csv', encoding='unicode_escape')
movies_revenue['IsAnimation'] = np.nan

for i, mov in movies_revenue.iterrows():
    if mov['ActorsAvg'] > 0:
        movies_revenue.at[i, 'IsAnimation'] = 1
    else:
        movies_revenue.at[i, 'IsAnimation'] = 0

movies_revenue.to_csv('datasets/movies-revenue.csv')
