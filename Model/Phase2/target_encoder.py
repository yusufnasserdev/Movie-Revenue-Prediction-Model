import time
import warnings
import category_encoders as ce
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn import tree

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# Loading Data![](C:/Users/s/AppData/Local/Temp/download.jpg)
movies = pd.read_csv('datasets/movies-revenue-classification.csv')

# Dividing Data
X = movies
Y = X['MovieSuccessLevel']

# Preparing Data
X.drop(['movie_title', 'director'], axis=1, inplace=True)
Y = np.where(Y == "S", 5, np.where(Y == "A", 4, np.where(Y == "B", 3, np.where(Y == "C", 2, 1))))
X['MovieSuccessLevel'] = Y
X = pd.DataFrame(X, columns=movies.keys())

tenc = ce.TargetEncoder()

df_city = tenc.fit_transform(movies['director'], movies['MovieSuccessLevel'])
df_city.to_csv("out.csv")
print(df_city)
