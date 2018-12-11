# -*- coding: utf-8 -*-
"""
Created on Mon May 21 02:06:17 2018

@author: Vaibhav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

Y_train = train.iloc[:,-1:].values

train.drop(['Id','Alley','Street','Utilities', 'PoolQC', 'Fence', 'MiscFeature','FireplaceQu','SalePrice'], 1, inplace = True)
test.drop(['Id','Alley','Street','Utilities', 'PoolQC', 'Fence', 'MiscFeature','FireplaceQu'], 1, inplace = True)

combined_data = train.append(test)

combined_data['MSSubClass'] = combined_data.MSSubClass.astype(dtype = object, errors = 'ignore')
combined_data['MoSold'] = combined_data.MoSold.astype(dtype = object, errors = 'ignore')
combined_data['YrSold'] = combined_data.YrSold.astype(dtype = object, errors = 'ignore')

def plot_missing_values(dataset):
    nan_values = dataset.isnull().sum()
    nan_values = nan_values[nan_values>0]
    nan_values.sort_values(inplace = True)
#    nan_values.plot.bar()
    return nan_values

missing_values = plot_missing_values(combined_data)

nan_indexes = list(missing_values.index)

def fill_missing_values(dataset,missing_values_indexes):
    for item in missing_values_indexes:
        if(dataset.dtypes[item] != 'object'):
            dataset[item].fillna(dataset[item].mean(), inplace = True)
        else:
 #           dataset[item].fillna(dataset[item].mode()[0], inplace = True)
            dataset[item].fillna("MISSING", inplace = True)

fill_missing_values(combined_data,nan_indexes)

''' #For Future reference (observing and filling the values)
train['LotFrontage'].plot.bar()
sns.distplot(train['LotFrontage'])
train['LotFrontage'].fillna(train['LotFrontage'].mean(), inplace = True)

count_each = Counter(train['GarageCond'])
train['GarageCond'].fillna(train['GarageCond'].mode()[0], inplace = True)
'''

# Encoding categorical data
def encode_categorical(dataset):
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder= LabelEncoder()
    qualitative = [q for q in dataset.columns if dataset.dtypes[q] == 'object']
    list_of_categorical_indexes = []
    list_of_categorical_counts = []
    dataset_columns = list(dataset.columns)
    
    for item in qualitative:
        dataset[item] = labelencoder.fit_transform(dataset[item])
        list_of_categorical_indexes.append(dataset_columns.index(item))
        list_of_categorical_counts.append(set(dataset[item]).__len__())
    
    onehotencoder = OneHotEncoder(categorical_features = list_of_categorical_indexes)    
    dataset = onehotencoder.fit_transform(dataset).toarray()
    dataset = pd.DataFrame(dataset)
    
    # Avoiding Dummy trap
    modified_i = 0
    for i in list_of_categorical_counts:
        dataset.drop([modified_i], 1, inplace = True)
        modified_i += i
    return dataset

combined_data = encode_categorical(combined_data)
#temp = pd.read_csv('train.csv')

X_train = combined_data.iloc[:1460, :].values

X_test = combined_data.iloc[1460:,:].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
Y_train = sc_y.fit_transform(Y_train)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 42)
regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)

# K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = Y_train, cv = 10)

# Grid Search
from sklearn.model_selection import GridSearchCV
parameters = [ {'n_estimators' : [50,100,150,200], 'min_samples_leaf' : [2,5,10,15,20]}]
grid_search = GridSearchCV(estimator = regressor, param_grid = parameters, scoring = 'mean_squared_error', cv = 10)
grid_search = grid_search.fit(X_train, Y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


Y_pred = sc_y.inverse_transform(Y_pred)
Y_pred = pd.DataFrame(Y_pred)
Y_pred.to_csv('House_Price_RandomForest_9.0.csv', sep=',')
