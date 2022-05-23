import pickle
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")

# Loading Data
movies = pd.read_csv('datasets/movies-revenue.csv')

# Dividing Data
X = movies
Y = X['revenue']


# Preparing Data
X.drop(['movie_title', 'director', 'ActorsAvg'], axis=1, inplace=True)
X = pd.DataFrame(X, columns=movies.keys())

# Top Features Extraction

# Getting Correlation
corr = X.corr()

# Extracting top feature via correlation
top_feature = corr.index[abs(corr['revenue']) > 0.2]

# Plotting Correlation
plt.subplots(figsize=(12, 8))
top_corr = X[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

# Deleting other features
top_feature = top_feature.delete(top_feature.get_loc('revenue'))
X = X[top_feature]

X.to_csv('datasets/out.csv', index=False)

# Data Splits
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=20, test_size=0.2, shuffle=True)

# Feature Scaling
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Polynomial Model
poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)

poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, Y_train)

print(X_train_poly)

pickle.dump(poly_model, open('poly_regression.pkl', 'wb'))

Y_test_predicted = poly_model.predict(poly_features.transform(X_test))
y_train_predicted = poly_model.predict(X_train_poly)

train_err = metrics.mean_squared_error(Y_train, y_train_predicted)
test_err = metrics.mean_squared_error(Y_test, Y_test_predicted)

print('Train subset (MSE) for degree {}: ', train_err)
print('Mean Square Error', test_err)
