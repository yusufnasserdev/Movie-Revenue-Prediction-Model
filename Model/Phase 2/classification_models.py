import warnings

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import metrics
from sklearn import tree

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

# Loading Data
movies = pd.read_csv('datasets/movies-revenue-classification.csv')

# Dividing Data
X = movies
Y = X['MovieSuccessLevel']

# Preparing Data
X.drop(['movie_title', 'director'], axis=1, inplace=True)
Y = np.where(Y == "S", 5, np.where(Y == "A", 4, np.where(Y == "B", 3, np.where(Y == "C", 2, 1))))
X['MovieSuccessLevel'] = Y
X = pd.DataFrame(X, columns=movies.keys())

"""
# Top Features Extraction
corr = X.corr()
print(corr)
top_feature = corr.index[abs(corr['MovieSuccessLevel']) > 0.2]
plt.subplots(figsize=(12, 8))
top_corr = X[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
top_feature = top_feature.delete(top_feature.get_loc('MovieSuccessLevel'))
X = X[top_feature]
"""

# Data Splits
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=20, test_size=0.2, shuffle=True)

# Feature Scaling
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------- #

# Hyper Parameters

C = 1.3  # SVM regularization parameter
m_degree = 7

# ---------------------------------------------------------- #

# Poly Model
poly_model = svm.SVC(kernel='poly', degree=m_degree, C=C).fit(X_train, Y_train)

print("Accuracy Poly:", metrics.accuracy_score(Y_test, poly_model.predict(X_test)))
print('R2 Score', metrics.r2_score(Y_test, poly_model.predict(X_test)))
print('Mean Square Error', metrics.mean_squared_error(Y_test, poly_model.predict(X_test)), '\n')

# ---------------------------------------------------------- #

# Linear Model
linear_svc = svm.LinearSVC(C=C).fit(X_train, Y_train)
# s = OneVsOneClassifier(lin_svc).fit()

print("Accuracy linear:", metrics.accuracy_score(Y_test, linear_svc.predict(X_test)))
print('R2 Score', metrics.r2_score(Y_test, linear_svc.predict(X_test)))
print('Mean Square Error', metrics.mean_squared_error(Y_test, linear_svc.predict(X_test)), '\n')

# ---------------------------------------------------------- #

# Rbf Model
rbf_svc = svm.SVC(kernel='rbf', C=C).fit(X_train, Y_train)
# s = OneVsOneClassifier(svc).fit(X_train, Y_train)

print("Accuracy rbf:", metrics.accuracy_score(Y_test, rbf_svc.predict(X_test)))
print('R2 Score', metrics.r2_score(Y_test, rbf_svc.predict(X_test)))
print('Mean Square Error', metrics.mean_squared_error(Y_test, rbf_svc.predict(X_test)), '\n')

# ---------------------------------------------------------- #

# Linear Kernel Model
linear_kernel_svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)

linear_kernel_prediction = linear_kernel_svc.predict(X_test)

print("Accuracy Linear kernel:", metrics.accuracy_score(Y_test, linear_kernel_prediction))
print('R2 Score', metrics.r2_score(Y_test, linear_kernel_prediction))
print('Mean Square Error', metrics.mean_squared_error(Y_test, linear_kernel_prediction), '\n')

# ---------------------------------------------------------- #

# Logistic Regression Model
logistic_regression_model = LogisticRegression(random_state=0).fit(X_train, Y_train)

print("Accuracy Logistic Regression:", metrics.accuracy_score(Y_test, logistic_regression_model.predict(X_test)))
print('R2 Score', metrics.r2_score(Y_test, logistic_regression_model.predict(X_test)))
print('Mean Square Error', metrics.mean_squared_error(Y_test, logistic_regression_model.predict(X_test)), '\n')

# ---------------------------------------------------------- #

# Decision Tree Model

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)

print("Accuracy Decision Tree:", metrics.accuracy_score(Y_test, poly_model.predict(X_test)))
print('R2 Score', metrics.r2_score(Y_test, poly_model.predict(X_test)))
print('Mean Square Error', metrics.mean_squared_error(Y_test, poly_model.predict(X_test)), '\n')
