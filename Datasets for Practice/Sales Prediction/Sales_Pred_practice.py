# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 20:07:56 2018

@author: bkn
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("train_set.csv")
test = pd.read_csv("test_set.csv")

print(train.describe())

sns.distplot(train['Item_Outlet_Sales'])
train['Item_Outlet_Sales'].plot.box()
train['Item_Fat_Content'].value_counts(normalize = True).plot.bar()

train.groupby('Item_Fat_Content').size()
sns.distplot(train.groupby('Item_Fat_Content').size())

outlet_size = pd.pivot_table(data = train, values = 'Item_Outlet_Sales', index = 'Item_Fat_Content', aggfunc = [sum,np.mean] )

plt.bar(outlet_size.index, outlet_size[outlet_size.columns[1]])

corr_mat = train.corr()
sns.heatmap(corr_mat)

train.isnull().sum()
train.isnotnull().sum()

train.drop('Item_Identifier', axis = 1, inplace = True)
train['Outlet_Size'].fillna(train['Outlet_Size'].mode()[0], inplace = True)

train.hist()

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
labelencoder.fit_transform(train['Outlet_Size'])
onehotencoder = OneHotEncoder(categorical_features = 'Outlet_Size')
train = onehotencoder.fit_transform(train).toarray()
train = pd.DataFrame(train)

from sklearn.preprocessing import StandardScalar
sc = StandardScalar()
sc_x = sc.fit_transform(train)



