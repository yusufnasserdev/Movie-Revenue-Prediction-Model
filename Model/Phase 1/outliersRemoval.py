import pandas as pd

movies = pd.read_csv('datasets/movies-revenue.csv')
q_low = movies.quantile(0.25)
q_hi = movies.quantile(0.75)
IQR = q_hi - q_low
output = (movies < (q_low - 1.5 * IQR)) | (movies > (q_hi + 1.5 * IQR))
output.to_csv('datasets/movies-revenue.csv')
