import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


def model_trial(x_train, x_test, y_traine, y_test, model, degree=30):
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(x_train)

    model.fit(X_train_poly, y_traine)

    y_train_predicted = model.predict(X_train_poly)
    prediction = model.predict(poly_features.fit_transform(x_test))

    train_err = metrics.mean_squared_error(y_train, y_train_predicted)
    test_err = metrics.mean_squared_error(y_test, prediction)
    # print('Train subset (MSE) for degree {}: '.format(degree), train_err)
    print('Test subset (MSE) for degree {}: '.format(degree), test_err)


# Load players data
data = pd.read_csv('datasets/movies-revenue.csv')
# Drop the rows that contain missing values
data.dropna(how='any', inplace=True)

X = data  # Features
Y = data['revenue']  # Label

X.drop(['title', 'director'], axis=1, inplace=True)

# Feature Selection
# Get the correlation between the features
corr = data.corr()

# Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['revenue']) > 0.2]
top_feature = top_feature.delete(top_feature.get_loc('revenue'))
X = X[top_feature]

# Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=10)

degreee = 3

model_trial(X_train, X_test, y_train, y_test, linear_model.Ridge(), degreee)
model_trial(X_train, X_test, y_train, y_test, linear_model.Lasso(), degreee)
model_trial(X_train, X_test, y_train, y_test, linear_model.BayesianRidge(), degreee)
