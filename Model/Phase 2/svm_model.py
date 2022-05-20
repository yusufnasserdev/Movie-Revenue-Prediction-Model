import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import MinMaxScaler

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
#"""
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
#"""
# Data Splits
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=20, test_size=0.2, shuffle=True)

# Feature Scaling
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SVM Linear

C = 1.3  # SVM regularization parameter

polymodel = svm.SVC(kernel='poly', degree=7, C=0.01).fit(X_train, Y_train)
predictions = polymodel.predict(X_train)
trainacc = np.mean(predictions == Y_train)
testacc = polymodel.score(X_test, Y_test)
print("Training Poly model acc = ", trainacc)
print("Testing Poly model acc = ", testacc, "\n")


lin_svc = svm.LinearSVC(C=C).fit(X_train, Y_train)
prediction = lin_svc.predict(X_train)
# s = OneVsOneClassifier(lin_svc).fit()

trainacc = np.mean(prediction == Y_train)
testacc = lin_svc.score(X_test, Y_test)
print("Training Linear model acc= ", trainacc)
print("Testing linear model accurcy = ", testacc, "\n")


C = 1.3 # SVM regularization parameter
svc = svm.SVC(kernel='rbf', C=C)
s=OneVsOneClassifier(svc).fit(X_train, Y_train)
print('training',s.score(X_train,Y_train))
print('testing',s.score(X_test,Y_test))

predictions = svc.predict(X_test)
# accuracy = np.mean(predictions == ytest)
# print('Testing accuracy :', accuracy)
# print('-----------------------------')
print(svc.score(X_test,Y_test))


