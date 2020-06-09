# Implementation of Linear regression on SAT-GPA database
# Linear Regression is used to predict values that follow a linear trend

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split 

df = pd.read_csv('SAT_GPA.csv')

X = df['GPA']
Y = df['SAT']

# Convert to numpy arrays and make them into n x 1 matrices to feed into 
# sklearn model
X = X.values.reshape(-1, 1)
y = Y.values.reshape(-1, 1)

# Split the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

lgr = linear_model.LinearRegression()

lgr.fit(X_train, y_train)

y_pred = lgr.predict(X_test)

# Plot the data and the predicted values
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred, color = 'red')
plt.show()

# score is the r2 score used to evaluate the model. Poor r2 score
# (closer to 0 ) indicates poor prediction
print(lgr.score(X_test, y_test))
