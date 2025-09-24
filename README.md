# MLCIE



## Program 1: Basic Linear Regression with NumPy
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


# Program 2: Linear Regression with Manually Entered Data
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    'YearsExperience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7, 3.9, 4.0, 4.0, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0, 6.8, 7.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3],
    'Salary': [39343, 46205, 37731, 43525, 39891, 56957, 60150, 54445, 64445, 57189, 63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81368, 93940, 91738, 98273, 101302, 113812, 109431, 105582, 113969, 112635]
}
dataset = pd.DataFrame(data)

X = dataset[['YearsExperience']]
y = dataset['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Intercept:", model.intercept_)
print("Coefficient (Slope):", model.coef_[0])

comparison = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': y_pred.flatten()})
print("\nComparison of Actual vs Predicted Salaries:")
print(comparison)

plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Regression line')

plt.xlabel("Year of Experience")
plt.ylabel("Salary")
plt.title("Salary vs Experience (Linear Regression)")
plt.legend()
plt.show()



# Program 3: Linear Regression from a CSV File
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

file_path = "c:\\users\\SINCHANA\\anaconda\\salary_data.csv"
data = pd.read_csv(file_path)

print("First 5 rows of the dataset:")
print(data.head())

X = data[['YearsExperience']]
y = data['Salary']

model = LinearRegression()
model.fit(X, y)

model_coefficient = model.coef_[0]
model_intercept = model.intercept_

print("\nModel Coefficient (Slope):", model_coefficient)
print("Model Intercept:", model_intercept)

plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression line')

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()


# Progrm 4: Python program on logistic Regression 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = {
    'Hours_Studied': [1,2,3,4,5,6,7,8,9,10],
    'Sleep_Hours':  [8,7,6,6,5,5,4,3,2,2],
    'Pass':         [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)
X = df[['Hours_Studied', 'Sleep_Hours']]
y = df['Pass']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

new_data = np.array([[6, 5], [2, 7]])  
predictions = model.predict(new_data)
print("\nPredictions for new data:", predictions)


# Program 5: Using Matplot Lib

import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 8],
    [2, 7],
    [3, 6],
    [4, 6],
    [5, 5],
    [6, 5],
    [7, 4],
    [8, 3],
    [9, 2],
    [10, 2]
])
y = np.array([0,0,0,0,1,1,1,1,1,1])

X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X_norm = (X - X_mean) / X_std

m, n = X_norm.shape
weights = np.zeros(n)
bias = 0
lr = 0.1
epochs = 1000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for epoch in range(epochs):
    z = np.dot(X_norm, weights) + bias
    y_pred = sigmoid(z)
    
    loss = -(1/m) * np.sum(y*np.log(y_pred+1e-9) + (1-y)*np.log(1-y_pred+1e-9))
    
    dw = (1/m) * np.dot(X_norm.T, (y_pred - y))
    db = (1/m) * np.sum(y_pred - y)
    
    weights -= lr * dw
    bias -= lr * db

def predict(X):
    z = np.dot(X, weights) + bias
    return np.where(sigmoid(z) >= 0.5, 1, 0)

y_pred_final = predict(X_norm)
print("Final Accuracy:", np.mean(y_pred_final == y))

plt.figure(figsize=(8,6))

plt.scatter(X[y==0][:,0], X[y==0][:,1], color='red', label="Fail (0)")
plt.scatter(X[y==1][:,0], X[y==1][:,1], color='green', label="Pass (1)")

x_values = np.linspace(min(X[:,0])-1, max(X[:,0])+1, 100)
x_norm = (x_values - X_mean[0]) / X_std[0]

if weights[1] != 0:  # avoid divide by zero
    y_values = -(weights[0]*x_norm + bias)/weights[1]
    y_values = y_values * X_std[1] + X_mean[1]
    plt.plot(x_values, y_values, color="blue", linewidth=2, label="Decision Boundary")

plt.xlabel("Hours Studied")
plt.ylabel("Sleep Hours")
plt.legend()
plt.title("Logistic Regression Decision Boundary (from scratch)")
plt.show()