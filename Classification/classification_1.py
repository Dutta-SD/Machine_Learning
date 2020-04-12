# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:59:29 2020

@author: sandip
"""
# =============================================================================
#          Classification of patients based on Various Parameters
# =============================================================================

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("diabetes.csv")

# Information about the patients
X = data.iloc[:, :-1].values

# Outcomes
y = data.iloc[:, -1:].values.flatten()

# =============================================================================
#                     *** Data Exploration ***
# =============================================================================

# To check the dataframe
# =============================================================================
# print(data.head())
# print(data.describe())
# =============================================================================

# To check if null values or not
# =============================================================================
# is_value_null = False
# for i in data.columns.values:
#     if data[i].isnull().values.any():
#         is_value_null = True
#         break
# print(is_value_null)   ## Outputs to False, so no null values in dataset
# =============================================================================

# To visualise the data and see
# =============================================================================
# for i in range(0, 8):
#     plt.figure(i )
#     plt.scatter(X[:, i:i+1], y)
#     plt.xlabel(data.columns[i])
#     plt.ylabel("Is Diabetic")     
# =============================================================================

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 40, test_size = 0.20)

# Scale the attributes

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Regression Model training

reg_model = LogisticRegression(multi_class = "multinomial", solver = "lbfgs", C = 100)
reg_model.fit(X_train, y_train)

# Predict the values

y_pred = reg_model.predict(X_test).flatten()

# Calculate goodness of model

from sklearn import metrics
print("Accuracy Score: ", metrics.accuracy_score(y_test, y_pred))

#null accuracy
print("Null Model Score: ", max(y_test.mean(), 1 - y_test.mean()))
