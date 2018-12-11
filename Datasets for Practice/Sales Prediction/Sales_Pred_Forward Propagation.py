# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 15:22:56 2018

@author: bkn
"""
import pandas as pd
import numpy as np
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")

# Load dataset
train = pd.read_csv('train_set.csv')
test = pd.read_csv('test_set.csv')
train_original=train.copy()
test_original=test.copy()

Y_train = train.iloc[:,-1:]

train.drop(['Item_Identifier','Outlet_Identifier', 'Item_Outlet_Sales'], 1, inplace = True)
test.drop(['Item_Identifier','Outlet_Identifier'], 1, inplace = True)
combined_data = train.append(test)

combined_data['Item_Fat_Content'].replace('LF', 'Low Fat',inplace=True)
combined_data['Item_Fat_Content'].replace('low fat', 'Low Fat',inplace=True)
combined_data['Item_Fat_Content'].replace('reg', 'Regular',inplace=True)
combined_data['Outlet_Size'].fillna(combined_data['Outlet_Size'].mode()[0], inplace=True)
combined_data['Item_Weight'].fillna(combined_data['Item_Weight'].mean(), inplace=True)

combined_data.isnull().sum()
#combined_data['Outlet_Establishment_Year'] = combined_data.Outlet_Establishment_Year.astype(dtype = object, errors = 'ignore')  
combined_data = combined_data[['Item_Fat_Content','Item_MRP','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type']]  #Best Selection (p<0.05, AdjR=5.63)
#combined_data = combined_data[['Item_Weight','Item_Fat_Content','Item_MRP','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type']]  ( AdjR=5.63)
#combined_data = combined_data[['Item_Weight','Item_MRP','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type']] #Best Ranking ( AdjR=5.62)
#combined_data = combined_data[['Item_Weight','Item_Visibility','Item_MRP','Outlet_Establishment_Year','Outlet_Size','Outlet_Location_Type','Outlet_Type']] # Good Enough
#combined_data = combined_data[['Item_MRP','Outlet_Type']]

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

combined_data = np.append(arr = np.ones((14204,1)).astype(int), values = combined_data, axis = 1)
combined_data = pd.DataFrame(combined_data)
combined_data[0] = pd.to_numeric(combined_data[0])

X_train = combined_data.iloc[:8523, :].values
X_test = combined_data.iloc[8523:,:].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train[:, 1:] = sc_X.fit_transform(X_train[:, 1:])
X_test[:, 1:] = sc_X.transform(X_test[:, 1:])
sc_y = StandardScaler()
Y_train = sc_y.fit_transform(Y_train)


##### Backward Elimination
import statsmodels.formula.api as sm

regressor_OLS = sm.OLS(endog = Y_train, exog = X_train).fit()
print(regressor_OLS.summary())


# Fitting Random Forest Regression to the dataset
#from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators = 100, random_state = 42)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = regressor.predict(X_test)
Y_pred = sc_y.inverse_transform(Y_pred)
Y_pred = pd.DataFrame(Y_pred)

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



