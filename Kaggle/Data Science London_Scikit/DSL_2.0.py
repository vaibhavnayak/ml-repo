# -*- coding: utf-8 -*-
"""
Created on Thu May 24 06:01:56 2018

@author: Vaibhav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

train = pd.read_csv('train.csv',header=None)
test = pd.read_csv('test.csv',header=None)
trainLabels = pd.read_csv('trainLabels.csv',header=None)

""" Check for Missing values
def plot_missing_values(dataset):
    nan_values = dataset.isnull().sum()
    nan_values = nan_values[nan_values>0]
    nan_values.sort_values(inplace = True)
    nan_values.plot.bar()
    return nan_values

missing_values = plot_missing_values(train)
missing_values = plot_missing_values(test)
missing_values = plot_missing_values(trainLabels
"""

X_train = train
X_test = test
Y_train = trainLabels

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(C = 3, kernel = 'rbf', random_state = 0, gamma = 0.054, decision_function_shape ='ovo', degree = 1)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

# K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies= cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [ {'C':list(np.arange(1.0, 10.0, 1.0)),'degree':list(np.arange(1, 10, 1)), 'kernel' : ['rbf'], 'gamma' :[0.054], 'decision_function_shape' : ['ovr']}]
## Do not include 'precomputed' in the list of kernals, it can only be used for square matrices
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 10)
grid_search = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

Y_pred = pd.DataFrame(Y_pred)
Y_pred.index += 1                       # Pre-defining Index Value, increase by 1
Y_pred.to_csv('DSL_KSVM_4.0.csv', sep=',', header=['Solution'], index_label = 'Id')

