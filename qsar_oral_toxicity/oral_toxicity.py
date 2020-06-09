# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 13:36:18 2020

@author: sandip
"""

# Oral Toxicity Prediction
# Dataset taken from: 
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the Sample
df = pd.read_csv('qsar_oral_toxicity.csv', sep=';', header=None)
# Separate independent and Dependent Features
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# We saw that it was an imbalanced DataSet
# To fix it we used SMOTE and increased the number of samples
from imblearn.combine import SMOTETomek

smk = SMOTETomek()
X, y = smk.fit_sample(X, y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Label Encode the predictions columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Scale the Data
from sklearn.preprocessing import StandardScaler
nm = StandardScaler()
X_train = nm.fit_transform(X_train)
X_test = nm.transform(X_test)

# The Dataset has about 1000 columns which is of very high dimensions
# So we reduce it by PCA
from sklearn.decomposition import PCA

pc = PCA(0.95, whiten=True)
X_train = pc.fit_transform(X_train)
X_test = pc.transform(X_test)

# Applying Simple Logistic Regression Model
from sklearn.linear_model import LogisticRegression
lR = LogisticRegression(max_iter = 100, n_jobs=8, solver='saga')
lR.fit(X_train, y_train)
y_preds  = lR.predict(X_test)

# Predicting The Output
from sklearn.metrics import f1_score

# The score comes to about 0.93
print(f1_score(y_test, y_preds))