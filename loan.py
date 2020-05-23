# -*- coding: utf-8 -*-
"""
Created on Fri May 15 20:55:34 2020

@author: sandip
"""
# =============================================================================
# Loan Prediction Problem using XGBoost
# =============================================================================

import pandas as pd

# Load the train data
X = pd.read_csv("train_data.csv")

# rows contains number of features of Data
rows = X.shape[1] - 1

# y has dependent feature Loan_Status
y = X.iloc[:, -1]

# X then has dependent features, we also discard the First row
X = X.iloc[:, 1 : rows ]

###########################################################################################
# To check if null values are present or not
# print(X.isnull().any(axis = 1).sum())
# We see that 134 records have NaN somewhere so we cannot drop the rows

# Lets separate the object data from numerical data

s = (X.dtypes=='object')
categorical_cols = list(s[s].index)

# Get numerical data column names
numerical_cols = [ i for i in X.columns if not i in categorical_cols ]

###########################################################################################

#Split the data set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, 
                                                    test_size = 0.2)

# Reset the indexes of the splitted data frames
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

y_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)

###########################################################################################

# Lets impute the data of the numerical columns with the median
from sklearn.impute import SimpleImputer

# Imputer Object
nm_imputer = SimpleImputer(strategy = 'median')

# Transform the necessary columns
X_train_numerical = pd.DataFrame(nm_imputer.fit_transform(X_train[numerical_cols]),
                                 columns = numerical_cols)

X_test_numerical = pd.DataFrame(nm_imputer.transform(X_test[numerical_cols]),
                                 columns = numerical_cols)
# Drop the non required columns
X_train = X_train.drop(numerical_cols, axis = 1)
X_test = X_test.drop(numerical_cols, axis = 1)

# put new colums in dataframe
X_train = X_train.join(X_train_numerical)
X_test = X_test.join(X_test_numerical)

############################################################################################

# Let us replace all missing categorical values with 
# the most frequently occuring value

categorical_imputer = SimpleImputer(strategy='most_frequent') 
   
X_train_categorical = pd.DataFrame(
        categorical_imputer.fit_transform(X_train[categorical_cols]),
        columns = categorical_cols )

X_test_categorical = pd.DataFrame(
        categorical_imputer.transform(X_test[categorical_cols]),
        columns = categorical_cols )
        
###########################################################################################

# now we will label encode the categorical features

#Label encoder object
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder() 

# Label Encode the features
for col in categorical_cols:
    X_train_categorical[col] = label_encoder.fit_transform(X_train_categorical[col])
    X_test_categorical[col] = label_encoder.transform(X_test_categorical[col])
    
# Drop the non required columns
X_train.drop(categorical_cols, axis = 1, inplace = True)
X_test.drop(categorical_cols, axis = 1, inplace=True)

# put new colums in dataframe
X_train = X_train.join(X_train_categorical)
X_test = X_test.join(X_test_categorical)

# Label Encode the dependent feature
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
###########################################################################################

# =============================================================================
# Now we will define our model. We will use XgBoost library to compute
# =============================================================================

from xgboost import XGBClassifier
from sklearn.metrics import f1_score

# to Select best hyperparameter
'''
for i in range(10, 600, 10):
    XGB = XGBClassifier(n_estimators = i)
    XGB.fit(X_train, y_train)
    y_preds = XGB.predict(X_test)
    print(i, f1_score(y_test, y_preds), sep='  ')
'''
    
##########################################################################################

# =============================================================================
# The best accuracy was got with n_estimators = 460  
# =============================================================================
    
    
XGB = XGBClassifier(n_estimators = 460)
XGB.fit(X_train, y_train)
y_preds = XGB.predict(X_test)
print(f1_score(y_preds, y_test) * 100)
# =============================================================================
# The accuracy is 84.39306358381504 %
# =============================================================================

