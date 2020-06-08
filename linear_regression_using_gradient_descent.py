import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing input data
data = pd.read_csv('data.csv')
X = data.iloc[:, 0]
y = data.iloc[:, 1]

# building the model
m = c = 0

L = 0.0001 # learning rate
epochs = 1000 # number of iterations to perform gradient descent

n = float(len(X)) # number of elements in X

# performing gradient descent
for i in range(epochs):
	y_pred = m*X + c 	# current predicted value of Y
	D_m = (-2/n) * sum(X * (y - y_pred)) 	# derivate wrt m
	D_c = (-2/n) * sum(y - y_pred) 	# derivate wrt c
	m = m - L * D_m 	# update m
	c = c - L * D_c 	# update c

print(m, c)

# making predictions
y_pred = m*X + c

plt.scatter(X, y)
plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color='red') # regression line
plt.show()