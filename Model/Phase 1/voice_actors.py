import pandas as pd

actors = pd.read_csv('datasets/movie-voice-actors.csv')
actors.sort_values(['movie'], axis=0, inplace=True)

actors.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
actors.to_csv('datasets/movie-voice-actors.csv')
