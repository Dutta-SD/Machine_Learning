# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:55:34 2020

@author: sandip
"""
# =============================================================================
# Loan Prediction Problem
# =============================================================================

import pandas as pd

# Load the train data
X = pd.read_csv("train_data.csv")

# rows contains number of features of Data
rows = X.shape[1]

# y has dependent feature Loan_Status
y = X.iloc[:, -1]

# X then has dependent features, we also discard the First row
X = X.iloc[:, 1:rows]

#Split the data set
from sklearn.model_selection import train_test_split
X, test_X, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.2)

# See the data
# print(test_X.head())

# Describe the data
# print(X.describe())

# We see that record number 24 contains NaN values. So we make it as reference 
# To check whether All our Preprocessing steps are occuring properly or not
# We define a purpose for our needs

check_record = X.iloc[24]
def check_preprocessing():
    print( pd.concat( [check_record, X.iloc[24]], axis = 1, keys = ['original', 'modified']) )

# To check if null values are present or not
print(X.isnull().any(axis = 1).sum())

# We see that 134 records have NaN somewhere so we cannot drop the rows

# Lets separate the object data from numerical data

s = (X.dtypes=='object')
categorical_cols = list(s[s].index)

# Get numerical data column names
numerical_cols = [ i for i in X.columns if not i in categorical_cols ]

# print object columns
# print(object_cols)
# print(numerical_cols)

# Lets impute the data of the numerical columns with the median
from sklearn.impute import SimpleImputer
numerical_imputer = SimpleImputer(strategy='mean')

# All numerical Values extracted from DataFrame
numerical_X = X.select_dtypes(exclude = ['object'])
numerical_X_test = test_X.select_dtypes(exclude = ['object'])

# Fit the imputer with the median value
numerical_X = pd.DataFrame(numerical_imputer.fit_transform(numerical_X),columns = numerical_cols)
numerical_X_test = pd.DataFrame(numerical_imputer.transform(numerical_X_test),columns = numerical_cols)

# Replace the datas in the DataFrame
X[numerical_cols] = numerical_X
test_X[numerical_cols] = numerical_X_test
# Checks if preprocessing changes are correctly reflected or not
# check_preprocessing()

# All categorical values selected 
categorical_X = X.select_dtypes(include = ['object'])
categorical_X_test = test_X.select_dtypes(include = ['object'])

# Let us replace all missing categorical values with the most frequently occuring value
categorical_imputer = SimpleImputer(strategy='most_frequent')
categorical_X = pd.DataFrame(categorical_imputer.fit_transform(categorical_X),columns = categorical_cols)
categorical_X_test = pd.DataFrame(categorical_imputer.transform(categorical_X_test),columns = categorical_cols)

# Checks if preprocessing changes are correctly reflected or not
# check_preprocessing()

# Now we will label encode our categorical data
from sklearn.preprocessing import LabelEncoder
labelize = LabelEncoder()
for col in categorical_cols:
    categorical_X[col] = labelize.fit_transform(categorical_X[col])
    categorical_X_test[col] = labelize.transform(categorical_X_test[col])
    
#  Put data back inplace
X[categorical_cols] = categorical_X
test_X[categorical_cols] = categorical_X_test

#check_preprocessing()
# For the dependent feature
Label_Encoder = LabelEncoder()
y_train = Label_Encoder.fit_transform(y_train)
y_test = Label_Encoder.transform(y_test)

# =============================================================================
# Now we will define our model. We will use XgBoost library to compute
# =============================================================================

from xgboost import XGBClassifier
from sklearn.metrics import f1_score

'''
for i in range(10, 500, 10):
    XGB = XGBClassifier(n_estimators = i)
    XGB.fit(X, y_train)
    y_preds = XGB.predict(test_X)
    #print("n_iterations: ", i, " " , f1_score(y_preds, y_test) * 100)
'''
    
# =============================================================================
# We see that about 60 iterations the accuracy goes to maximum
# =============================================================================
    
XGB = XGBClassifier(n_estimators = 60)
XGB.fit(X, y_train)
y_preds = XGB.predict(test_X)
print(f1_score(y_preds, y_test) * 100)
# =============================================================================
# The accuracy is 78.97435897435898 %
# =============================================================================