# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 20:21:14 2018

@author: bkn
"""


# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
train = pd.read_csv('train_set.csv')
test = pd.read_csv('test_set.csv')

train_original=train.copy()
test_original=test.copy()

plt.figure(1)
plt.subplot(221)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(10,5), title= 'Gender')

plt.subplot(222)
train['Married'].value_counts(normalize=True).plot.bar(title= 'Married')

plt.subplot(223)
train['Self_Employed'].value_counts(normalize=True).plot.bar(title= 'Self_Employed')

plt.subplot(224)
train['Credit_History'].value_counts(normalize=True).plot.bar(title= 'Credit_History')

plt.show()

Y_train = train.iloc[:,-1:]

train.drop(['Loan_ID', 'Loan_Status'], 1, inplace = True)
test.drop(['Loan_ID'], 1, inplace = True)

combined_data = train.append(test)
combined_data['Dependents'] = combined_data.Dependents.astype(dtype = object, errors = 'ignore')
#combined_data['Loan_Amount_Term'] = combined_data.Loan_Amount_Term.astype(dtype = object, errors = 'ignore')
combined_data['Credit_History'] = combined_data.Credit_History.astype(dtype = object, errors = 'ignore')

def plot_missing_values(dataset):
    nan_values = dataset.isnull().sum()
    nan_values = nan_values[nan_values>0]
    nan_values.sort_values(inplace = True)
    nan_values.plot.bar()
    return nan_values

missing_values = plot_missing_values(combined_data)

nan_indexes = list(missing_values.index)

# box and whisker plots
combined_data.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)
plt.show()

# histograms
combined_data.hist()
plt.show()

# scatter plot matrix
scatter_matrix(combined_data)
plt.show()

def fill_missing_values(dataset,missing_values_indexes):
    for item in missing_values_indexes:
        if(dataset.dtypes[item] != 'object'):
            dataset[item].fillna(dataset[item].mean(), inplace = True)
        else:
            dataset[item].fillna(dataset[item].mode()[0], inplace = True)
#            dataset[item].fillna("MISSING", inplace = True)

fill_missing_values(combined_data,nan_indexes)

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

X_train = combined_data.iloc[:614, :].values
X_test = combined_data.iloc[614:,:].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#Y_train = sc_y.fit_transform(Y_train)

# Test options and evaluation metric
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('RF', RandomForestClassifier(n_estimators = 100, criterion = 'entropy')))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel = 'rbf')))

results = []
names = []
score_comparison = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=5, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    score_comparison.append(msg)
	
print(score_comparison)
    
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

## Fitting Random Forest Classifier to the dataset
#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier(n_estimators = 100, random_state = 42)
#classifier.fit(X_train, Y_train)

# Fitting SV Classifier to the dataset
classifier = SVC(random_state = 0)
classifier.fit(X_train, Y_train)

# Predicting the Test set results
Y_pred = classifier.predict(X_test)


# K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = 10)

Y_pred = pd.DataFrame(Y_pred)
Y_pred.to_csv('Loan_Approval_Prediction_v2.csv', sep=',')




