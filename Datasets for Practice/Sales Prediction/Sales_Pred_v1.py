# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 22:52:01 2018

@author: bkn
"""

"""
Hypothesis:
    Directly Proportional to sales - Item_Visibility



"""


import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")

# Load dataset
train = pd.read_csv('train_set.csv')
test = pd.read_csv('test_set.csv')

train_original=train.copy()
test_original=test.copy()

print(train.describe())

#### UNIVARIATE ANALYSIS

plt.figure(1)
plt.subplot(121)
sns.distplot(train['Item_Outlet_Sales']);
plt.subplot(122)
train['Item_Outlet_Sales'].plot.box(figsize=(9,3))
plt.show()

plt.figure(1)
plt.subplot(121)
sns.distplot(train['Item_MRP']);
plt.subplot(122)
train['Item_MRP'].plot.box(figsize=(9,3))
plt.show()

plt.figure(1)
plt.subplot(121)
sns.distplot(train['Item_Visibility']);
plt.subplot(122)
train['Item_Visibility'].plot.box(figsize=(9,3))
plt.show()

plt.figure(1)
plt.subplot(121)
sns.distplot(train['Item_Weight']);
plt.subplot(122)
train['Item_Weight'].plot.box(figsize=(9,3))
plt.show()


plt.figure(1)
plt.subplot(221)
train['Item_Identifier'].value_counts(normalize=True).plot.bar(figsize=(10,5), title= 'Item_Identifier')

plt.subplot(222)
train['Item_Fat_Content'].value_counts(normalize=True).plot.bar(title= 'Item_Fat_Content')

plt.subplot(222)
train['Item_Type'].value_counts(normalize=True).plot.bar(title= 'Item_Type')

plt.subplot(223)
train['Outlet_Identifier'].value_counts(normalize=True).plot.bar(title= 'Outlet_Identifier')

plt.subplot(224)
train['Outlet_Size'].value_counts(normalize=True).plot.bar(title= 'Outlet_Size')

plt.subplot(224)
train['Outlet_Location_Type'].value_counts(normalize=True).plot.bar(title= 'Outlet_Location_Type')

plt.subplot(224)
train['Outlet_Type'].value_counts(normalize=True).plot.bar(title= 'Outlet_Type')

plt.show()


# class distribution
print(train.groupby('Item_Identifier').size())
print(train.groupby('Item_Fat_Content').size())
print(train.groupby('Item_Type').size())
print(train.groupby('Outlet_Identifier').size())
print(train.groupby('Outlet_Size').size())
print(train.groupby('Outlet_Location_Type').size())
print(train.groupby('Outlet_Type').size())

plt.figure(1)
plt.subplot(121)
sns.distplot(train.groupby('Item_Identifier').size(), kde = False)


#### BIVARIATE ANALYSIS


Item_Fat_Content=pd.pivot_table(data = train, values = 'Item_Outlet_Sales', index = 'Item_Fat_Content', aggfunc=[sum,np.mean])
Outlet_Identifier=pd.pivot_table(data = train, values = 'Item_Outlet_Sales', index = 'Outlet_Identifier', aggfunc=[sum,np.mean])
Outlet_Size=pd.pivot_table(data = train, values = 'Item_Outlet_Sales', index = 'Outlet_Size', aggfunc=[sum,np.mean])
Outlet_Location_Type=pd.pivot_table(data = train, values = 'Item_Outlet_Sales', index = 'Outlet_Location_Type', aggfunc=[sum,np.mean])
Outlet_Type=pd.pivot_table(data = train, values = 'Item_Outlet_Sales', index = 'Outlet_Type', aggfunc=[sum,np.mean])

plt.bar(Item_Fat_Content.index, Item_Fat_Content[Item_Fat_Content.columns[1]])
plt.show()

plt.bar(Outlet_Identifier.index, Outlet_Identifier[Outlet_Identifier.columns[1]])
plt.show()

plt.bar(Outlet_Size.index, Outlet_Size[Outlet_Size.columns[1]])
plt.show()

plt.bar(Outlet_Location_Type.index, Outlet_Location_Type[Outlet_Location_Type.columns[1]])
plt.show()

plt.bar(Outlet_Type.index, Outlet_Type[Outlet_Type.columns[1]])
plt.show()


# box and whisker plots
train.plot(kind='box', subplots=True, layout=(3,2), sharex=False, sharey=False)
plt.show()

Outlet_Size__Outlet_Type=pd.crosstab(train['Outlet_Size'],train['Outlet_Type'])
Outlet_Size__Outlet_Type.div(Outlet_Size__Outlet_Type.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(5,3))
plt.show()

Outlet_Size__Outlet_Location_Type=pd.crosstab(train['Outlet_Size'],train['Outlet_Location_Type'])
Outlet_Size__Outlet_Location_Type.div(Outlet_Size__Outlet_Location_Type.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(5,3))
plt.show()

Outlet_Type__Outlet_Location_Type=pd.crosstab(train['Outlet_Type'],train['Outlet_Location_Type'])
Outlet_Type__Outlet_Location_Type.div(Outlet_Type__Outlet_Location_Type.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(5,3))
plt.show()

### Correlation between all the numeric variables
matrix = train.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");

# Modifications in the dataset


# Missing Value count
train.isnull().sum()

Y_train = train.iloc[:,-1:]

train.drop(['Item_Identifier','Outlet_Identifier', 'Item_Outlet_Sales'], 1, inplace = True)
test.drop(['Item_Identifier','Outlet_Identifier'], 1, inplace = True)

combined_data = train.append(test)
combined_data.isnull().sum()

combined_data['Item_Fat_Content'].replace('LF', 'Low Fat',inplace=True)
combined_data['Item_Fat_Content'].replace('low fat', 'Low Fat',inplace=True)
combined_data['Item_Fat_Content'].replace('reg', 'Regular',inplace=True)
combined_data['Outlet_Size'].fillna(combined_data['Outlet_Size'].mode()[0], inplace=True)
combined_data['Item_Weight'].fillna(combined_data['Item_Weight'].mean(), inplace=True)


# histograms
combined_data.hist()
plt.show()

# scatter plot matrix
scatter_matrix(combined_data)
plt.show()


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

X_train = combined_data.iloc[:8523, :].values
X_test = combined_data.iloc[8523:,:].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
Y_train = sc_y.fit_transform(Y_train)

# Spot Check Algorithms
models = []
#models.append(('LR', LinearRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('RF', RandomForestRegressor(n_estimators = 100)))
#models.append(('SVM', SVR(kernel = 'rbf')))
models.append(('Poly', PolynomialFeatures(degree = 4)))


# evaluate each model in turn
seed = 7
scoring = 'neg_mean_squared_error'
results = []
names = []
for name, model in models:
    if(name == "poly"):
        accuracies = cross_val_score(estimator = model, X = model.fit_transform(X_train), y = Y_train, cv = 10, scoring = scoring)
    else:
        accuracies = cross_val_score(estimator = model, X = X_train, y = Y_train, cv = 10, scoring = scoring)
    results.append(accuracies.mean())
    names.append(name)
    msg = "%s: %f (%f)" % (name, accuracies.mean(), accuracies.std())
    print(msg)

poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, Y_train)
Y_pred_poly = poly_reg.predict(X_test)


regressor = RandomForestRegressor(n_estimators = 100, random_state = 42)
regressor.fit(X_train, Y_train)

regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)

#accuracies = cross_val_score(estimator = regressor, X = X_train, y = Y_train, cv = 10)

Y_pred = sc_y.inverse_transform(Y_pred)

##### Backward Elimination
import statsmodels.formula.api as sm

X_opt = X_train[:,[0,1,2,3,4,5,6,7,8,9]]

X = np.append(arr = np.ones((8523,1)).astype(int), values = X_train, axis = 1)



regressor_OLS = sm.OLS(endog = Y_train, exog = X_opt).fit()
regressor_OLS.summary()


submission=pd.read_csv("SampleSubmission.csv")

submission['Item_Outlet_Sales']=Y_pred
submission['Item_Identifier']=test_original['Item_Identifier']
submission['Outlet_Identifier']=test_original['Outlet_Identifier']

pd.DataFrame(submission, columns=['Item_Identifier','Outlet_Identifier','Item_Outlet_Sales']).to_csv('Sales_Prediction_v1.csv', index=False)


## Compare Algorithms
#fig = plt.figure()
#fig.suptitle('Algorithm Comparison')
#ax = fig.add_subplot(111)
#plt.boxplot(results)
#ax.set_xticklabels(names)
#plt.show()



