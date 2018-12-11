# -*- coding: utf-8 -*-

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import math

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
passengerId = pd.read_csv('gender_submission.csv')
# Droping unwanted columns

train.drop(['PassengerId','Name','Ticket','Cabin'], 1, inplace = True)
test.drop(['PassengerId','Name','Ticket','Cabin'], 1, inplace = True)

#   train['Age'].fillna(train['Age'].mean(), inplace = True)
#   test['Age'].fillna(test['Age'].mean(), inplace = True)

# Handling missing values

train.fillna(train.mean(), inplace = True)
test.fillna(test.mean(), inplace = True)

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)
test['Embarked'].fillna(test['Embarked'].mode()[0], inplace = True)

X_train = train.iloc[:,1:].values
Y_train = train.iloc[:,0:1]

X_test = test.iloc[:,:].values
Y_test = passengerId.iloc[:,1:2]

# Encoding Categorical Variables

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_train = LabelEncoder()
X_train[:, 1] = labelencoder_X_train.fit_transform(X_train[:, 1])
X_train[:, 6] = labelencoder_X_train.fit_transform(X_train[:, 6])
onehotencoder = OneHotEncoder(categorical_features = [1,6])
X_train = onehotencoder.fit_transform(X_train).toarray()
X_train = pd.DataFrame(X_train)
X_train.drop([0,2],1,inplace = True)        # Avoiding Dummy trap

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_test = LabelEncoder()
X_test[:, 1] = labelencoder_X_test.fit_transform(X_test[:, 1])
X_test[:, 6] = labelencoder_X_test.fit_transform(X_test[:, 6])
onehotencoder2 = OneHotEncoder(categorical_features = [1,6])
X_test = onehotencoder2.fit_transform(X_test).toarray()
X_test = pd.DataFrame(X_test)
X_test.drop([0,2],1,inplace = True)         # Avoiding Dummy trap

#   Z = pd.DataFrame(X_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

y_pred = pd.DataFrame(y_pred)

y_pred[:,1:2] = y_pred.iloc[:,0:1]
y_pred[:,0:1] = passengerId.iloc[1:,0:1]
y_pred = pd.DataFrame(data = y_pred, columns=['PassengerId', 'Survived'])
y_pred.to_csv('Titanic Submission2.csv', sep=',')


