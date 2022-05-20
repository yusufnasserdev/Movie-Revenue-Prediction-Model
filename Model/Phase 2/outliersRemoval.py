import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def cap_data(df):
    for col in df.columns:
        if (((df[col].dtype)=='float64') | ((df[col].dtype)=='int64')):
            percentiles = df[col].quantile([0.25,0.75]).values
            df[col][df[col] <= percentiles[0]] = percentiles[0]
            df[col][df[col] >= percentiles[1]] = percentiles[1]
        else:
            df[col]=df[col]
    return df


movies = pd.read_csv('datasets/movies-revenue-classification.csv')
movies.drop(['movie_title', 'director'], axis=1, inplace=True)
movies = cap_data(movies)
movies.to_csv('out.csv')
