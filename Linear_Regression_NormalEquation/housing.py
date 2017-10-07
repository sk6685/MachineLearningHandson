# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 01:26:38 2017

@author: teqtron
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Housing.csv')
#print(dataset.head())
X = dataset.iloc[:, 2].values
print(X.shape)

X = X.reshape(len(X),1)
y = dataset.iloc[:, 1].values
y = y.reshape(len(y),1)

print(y.shape)

                
 # Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)


  # Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')

plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Price vs Lotsize  (Training set)')
plt.xlabel('Lotsize')
plt.ylabel('Price')
plt.show()
# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue',label="Predictions")

plt.title('Price vs Lotsize  (Test set)')
plt.xlabel('Lotsize')
plt.ylabel('Price')
plt.legend(loc="upper left", fontsize=14)

plt.show()

#to print individual prediction
print( str((regressor.predict(5000))) )


