from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns



iris = load_iris()
X = iris.data[:, [2, 3]]  # petal length & width
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

m=svm.SVC(kernel='linear')
m.fit(X_train, y_train)
y_pred = m.predict(X_test)

plt.plot(y_test, y_pred)
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("Accuracy vs predicted values")
plt.tight_layout()
plt.show()