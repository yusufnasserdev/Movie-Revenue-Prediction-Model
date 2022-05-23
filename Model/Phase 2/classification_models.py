import pickle
import time
import warnings

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

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
X.drop(['movie_title', 'director', 'ActorsAvg'], axis=1, inplace=True)
Y = np.where(Y == "S", 5, np.where(Y == "A", 4, np.where(Y == "B", 3, np.where(Y == "C", 2, 1))))
X['MovieSuccessLevel'] = Y
X = pd.DataFrame(X, columns=movies.keys())

"""
# Top Features Extraction

# Getting Correlation
corr = X.corr()

# Extracting top feature via correlation
top_feature = corr.index[abs(corr['MovieSuccessLevel']) > 0.2]

# Plotting Correlation
plt.subplots(figsize=(12, 8))
top_corr = X[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()

# Deleting other features
top_feature = top_feature.delete(top_feature.get_loc('MovieSuccessLevel'))
X = X[top_feature]
"""

# Dropping the label from the features dataframe
X.drop(['MovieSuccessLevel'], axis=1, inplace=True)


# Data Splits
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=20, test_size=0.2, shuffle=True)

# Feature Scaling
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------- #

# Hyper Parameters

C = 0.00001  # SVM regularization parameter
m_degree = 7

# ---------------------------------------------------------- #

# Poly Model
t0 = time.time()
poly_model = svm.SVC(kernel='poly', degree=m_degree, C=C).fit(X_train, Y_train)
print("Training time of poly_svc model:", time.time() - t0)
t0 = time.time()
p = poly_model.predict(X_test)
print("Testing time of poly_svc model:", time.time() - t0)
print("Accuracy Poly:", metrics.accuracy_score(Y_test, p))
print('R2 Score', metrics.r2_score(Y_test, p))
print('Mean Square Error', metrics.mean_squared_error(Y_test, p), '\n')

# ---------------------------------------------------------- #

# Linear Model
t0 = time.time()
linear_svc = svm.LinearSVC(C=C).fit(X_train, Y_train)
print("Training time of linear_svc model:", time.time() - t0)
# s = OneVsOneClassifier(lin_svc).fit()
t0 = time.time()
p = linear_svc.predict(X_test)
print("Testing time of linear_svc model:", time.time() - t0)
print("Accuracy linear:", metrics.accuracy_score(Y_test, p))
print('R2 Score', metrics.r2_score(Y_test, p))
print('Mean Square Error', metrics.mean_squared_error(Y_test, p), '\n')

# ---------------------------------------------------------- #

# Rbf Model
t0 = time.time()
rbf_svc = svm.SVC(kernel='rbf', C=C).fit(X_train, Y_train)
print("Training time of rbf_svc model:", time.time() - t0)
# s = OneVsOneClassifier(svc).fit(X_train, Y_train)
t0 = time.time()
p = rbf_svc.predict(X_test)
print("Testing time of rbf_svc model:", time.time() - t0)
print("Accuracy rbf:", metrics.accuracy_score(Y_test, p))
print('R2 Score', metrics.r2_score(Y_test, p))
print('Mean Square Error', metrics.mean_squared_error(Y_test, p), '\n')

# ---------------------------------------------------------- #

# Linear Kernel Model
t0 = time.time()
linear_kernel_svc = svm.SVC(kernel='linear', C=C).fit(X_train, Y_train)
pickle.dump(linear_kernel_svc, open('linearKernel.pkl', 'wb'))
print("Training time of linear_Kernel_svc model:", time.time() - t0)
t0 = time.time()
p = linear_kernel_svc.predict(X_test)
print("Testing time of linear_Kernel_svc model:", time.time() - t0)
print("Accuracy Linear kernel:", metrics.accuracy_score(Y_test, p))
print('R2 Score', metrics.r2_score(Y_test, p))
print('Mean Square Error', metrics.mean_squared_error(Y_test, p), '\n')

# ---------------------------------------------------------- #

# Logistic Regression Model
t0 = time.time()
logistic_regression_model = LogisticRegression(random_state=0).fit(X_train, Y_train)
pickle.dump(logistic_regression_model, open('logistic.pkl', 'wb'))
print("Training time of logistic_regression_model model:", time.time() - t0)
t0 = time.time()
p = logistic_regression_model.predict(X_test)
print("Testing time of logistic_regression_model model:", time.time() - t0)
print("Accuracy Logistic Regression:", metrics.accuracy_score(Y_test, p))
print('R2 Score', metrics.r2_score(Y_test, p))
print('Mean Square Error', metrics.mean_squared_error(Y_test, p), '\n')

# ---------------------------------------------------------- #

# Decision Tree Model

clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
t0 = time.time()
clf = clf.fit(X_train, Y_train)
pickle.dump(clf, open('DT.pkl', 'wb'))
print("Training time of DecisionTree_model model:", time.time() - t0)
t0 = time.time()
p = clf.predict(X_test)
print("Testing time of DecisionTree_model model:", time.time() - t0)
print("Accuracy Decision Tree:", metrics.accuracy_score(Y_test, p))
print('R2 Score', metrics.r2_score(Y_test, p))
print('Mean Square Error', metrics.mean_squared_error(Y_test, p), '\n')
