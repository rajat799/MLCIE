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

