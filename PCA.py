# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:12:02 2020

@author: sandip
"""

# =============================================================================
# To implement PCA in python with sklearn and check effectiveness
# =============================================================================

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

model = LogisticRegression(solver = 'lbfgs', max_iter = 2000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Without Dimensionality reduction: ", f1_score(y_test, y_pred))

pca = PCA(n_components=0.90)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("With dimensionality reduction: ", f1_score(y_test, y_pred))