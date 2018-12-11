# -*- coding: utf-8 -*-
"""
Created on Sun May 20 20:21:56 2018

@author: bkn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv('train.csv')

train.drop(['Id','Alley','Street','Utilities', 'PoolQC', 'Fence', 'MiscFeature'], 1, inplace = True)

quantitative = [q for q in train.columns if train.dtypes[q] != 'object']

quantitative.remove('Id')
quantitative.remove('SalePrice')

qualitative = [q for q in train.columns if train.dtypes[q] == 'object']

missing = train.isnull().sum()
missing = missing[missing>0]
missing.sort_values(inplace = True)
missing.plot.bar()

import seaborn as sns
import scipy.stats as st

y = train['SalePrice']

plt.figure(1)
plt.title('Johnson SU')
sns.distplot(y, kde = False, fit = st.johnsonsu)

plt.figure(2)
plt.title('Normal')
sns.distplot(y, kde = False, fit = st.norm)

plt.figure(3)
plt.title('Log Normal')
sns.distplot(y, kde = False, fit = st.lognorm)

import scipy.stats as stats

test_normality = lambda x:stats.shapiro(x.fillna(0))[1] <0.01
normal = pd.DataFrame(train[quantitative])
normal = normal.apply(test_normality)
print(not normal.any())

f = pd.melt(train, value_vars=quantitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")

