import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([2, 4, 5, 4, 5])

model = LinearRegression()
model.fit(X, Y)

slope = model.coef_[0]
intercept = model.intercept_

print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")

y_pred = model.predict([[6]])
print(f"Prediction for x=6: {y_pred[0]:.2f}")

plt.scatter(X, Y, color='blue', label='Data')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.legend()
plt.show()