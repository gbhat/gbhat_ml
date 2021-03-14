import sys

import matplotlib.pyplot as plt
import numpy as np
from mlfromscratch.utils import mean_squared_error
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


X, y = make_regression(n_samples=200, n_features=1, random_state=0, noise=15.0, bias=100.0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=10)

# Since the data is shuffled by train_test_split, we sort the test data to plot it neatly
y_test = y_test[X_test.flatten().argsort()]
X_test = np.sort(X_test, axis=0)

model = LinearRegression().fit(X_test, y_test)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_pred, y_test)
print("Mean Squared Error: ", mse)

plt.scatter(X_train, y_train, c='g', s=20)
plt.scatter(X_test, y_test, c='b', s=20)
plt.plot(X_test, y_pred, c='r', linewidth=2)
plt.show()
