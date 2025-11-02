# Program 3
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns, matplotlib.pyplot as plt

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
sc = StandardScaler(); Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)

knn = KNeighborsClassifier(n_neighbors=5).fit(Xtr, ytr)
yp = knn.predict(Xte)
print("Confusion Matrix:\n",(cm:=confusion_matrix(yte, yp)))
print("\nClassification Report:\n",classification_report(yte, yp, target_names=load_iris().target_names))
print("Accuracy:",accuracy_score(yte, yp))

sns.heatmap(cm, annot=True, cmap='Greens', xticklabels=load_iris().target_names, yticklabels=load_iris().target_names)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("KNN on Iris"); plt.show()
