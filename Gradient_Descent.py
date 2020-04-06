# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 09:50:39 2020

@author: sandip
"""

import numpy as np
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Concatentes a (100 x 1 ) array to the matrix X for calculations
X_b = np.c_[np.ones((100, 1)), X]

eta = 0.01        #learning rate
n_iterations = 1000
m = 100

theta = np.random.randn(2, 1)
# =============================================================================
# This portion is to implement Batch Gradient descent algorithm
# Batch Gradient Descent takes all the values and compuetes the gradient
# based on all the attributes. This is computationally expensive, but it is much
# more regular than other methods. 
# =============================================================================

for iteration in range(n_iterations):
     gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
     theta = theta - eta * gradients

print("Batch:", theta.flatten())

# =============================================================================
# This portion is to implement Stochastic Gradient descent. Stochastic 
# Gradient descent takes one attribute and then computes the gradient based on
# that attribute only. It's randomness is helpful to converge on the global
# minima. But it is very less regular and keeps on hovering near the minima.
# =============================================================================

n_epochs = 50 
t0, t1 = 5, 50    #learning schedule hyperparameters

def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index : random_index + 1]
        yi = y[random_index : random_index + 1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(epoch*m + i)
        theta = theta - eta * gradients
        
print("SGD: ", theta.flatten())

# =============================================================================
# SGD using sklearn's inbuilt SGDregressor class
# =============================================================================

from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter = 500, penalty = None, eta0 = 0.1, early_stopping = True)
sgd_reg.fit(X, y.ravel())

print(sgd_reg.intercept_, sgd_reg.coef_)
        