from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from Pre_processing import *

movies = pd.read_csv('datasets/movies-revenue.csv')
X = movies
Y = X['revenue']
X.drop(['title', 'director'], axis=1, inplace=True)

X = featureScaling(X, 0, 100)
X = pd.DataFrame(X, columns=movies.keys())
corr = X.corr()
top_feature = corr.index[abs(corr['revenue']) > 0.2]
plt.subplots(figsize=(12, 8))
top_corr = X[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(top_feature.get_loc('revenue'))
X = X[top_feature]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=20, test_size=0.2, shuffle=True)

poly_features = PolynomialFeatures(degree=3)
X_train_poly = poly_features.fit_transform(X_train)
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, Y_train)
Y_test_predicted = poly_model.predict(poly_features.transform(X_test))
prediction = poly_model.predict(poly_features.fit_transform(X_test))
y_train_predicted = poly_model.predict(X_train_poly)
train_err = metrics.mean_squared_error(Y_train, y_train_predicted)

print('Train subset (MSE) for degree {}: ', train_err)
print('Mean Square Error', metrics.mean_squared_error(Y_test, Y_test_predicted))
