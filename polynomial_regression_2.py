import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlfromscratch.utils import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# Load temperature data
data = pd.read_csv('TempLinkoping2016.csv')

time_X = np.atleast_2d(data["day"].values).T
temp = data["temperature"].values

X = time_X  # fraction of the year [0, 1]
y = temp
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=10)

# Since the data is shuffled by train_test_split, we sort the test data to plot it neatly
y_test = y_test[X_test.flatten().argsort()]
X_test = np.sort(X_test, axis=0)

poly = PolynomialFeatures(degree=5, include_bias=False)
poly_X_train = poly.fit_transform(X_train)
poly_X_test = poly.fit_transform(X_test)

model = LinearRegression()
model.fit(poly_X_train, y_train)
y_pred = model.predict(poly_X_test)
mse = mean_squared_error(y_pred, y_test)
print("Mean Squared Error: ", mse)

plt.scatter(X_train, y_train, c='g', s=20)
plt.scatter(X_test, y_test, c='b', s=20)
plt.plot(X_test, y_pred, linewidth=2, c='r')

plt.legend()
plt.show()
