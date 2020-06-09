# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 00:58:02 2020

@author: sandip
"""

# =============================================================================
# *** Parkinsons Disease Detection Using XGBoost Library
# =============================================================================

import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Read Data into Pandas DataFrame
p_data = pd.read_csv("parkinsons.data")
# Exclude Categorical Variables
p_data = p_data.select_dtypes(exclude = ['object'])

# Data Exploration : print(p_data.columns)

# Colums that contain data about the patients
cols = [x for x in p_data.columns if x != 'status']

# X is the feature matrix
X = p_data[cols]

# y is the target vector
y = p_data['status']

# Data Exploration: print(y.head())
# Data Exploration: print(y.describe())

# Split the Data into train test 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.2)

# Define the model
x_model = XGBClassifier(n_estimators = 2000, learning_rate = 0.01)

# Create a pipeline which with appropriate steps
x_pipeline = Pipeline(steps =[
        ('preprocessing', StandardScaler() ), 
        ('model', x_model)       
        ])

# fit the model
x_pipeline.fit(X_train, y_train)
#Get predictions
predicted_vals = x_pipeline.predict(X_test)

# Accuracy Score
print("Accuracy with this model: ", accuracy_score(y_test, predicted_vals))

# This score is pretty high. This indicates that there might be 
# a good chance of data leakage.
