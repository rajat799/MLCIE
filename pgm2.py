import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(x_train, y_train)

y_pred = svm_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("accuracy: ", accuracy)

print("\n confusion matrix: \n", confusion_matrix(y_test, y_pred))
print("\n classification report: \n", classification_report(y_test, y_pred))