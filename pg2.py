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
