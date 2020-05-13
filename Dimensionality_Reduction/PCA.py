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

# We implement a Logistic Regression model To check how much reduction in accuracy
# occurs due to PCA
model = LogisticRegression(solver = 'lbfgs', max_iter = 2000)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Without Dimensionality reduction: ", f1_score(y_test, y_pred))

# Now we apply PCA to see how much Reduction in F1 score do we get
# If we see that we get near about accuracy, then PCA can be used to
# compress data without much loss. This has the added benefit of making
# training times very less

pca = PCA(n_components=0.90)                #n_components 90% variance kept
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("With dimensionality reduction: ", f1_score(y_test, y_pred))

# A score of about 95% without PCA and 92% with PCA was obatined
# So we see that PCA is indeed useful
