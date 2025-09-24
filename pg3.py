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
