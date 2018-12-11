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

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)

Y_pred = pd.DataFrame(Y_pred)
Y_pred.index += 1                       # Pre-defining Index Value, increase by 1
Y_pred.to_csv('DSL_RF_3.0.csv', sep=',', header=['Solution'], index_label = 'Id')

