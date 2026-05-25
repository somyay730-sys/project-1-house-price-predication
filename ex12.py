import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from ex5 import y_train

# ----------------------------------
# Load Dataset
# ----------------------------------
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# ----------------------------------
# Train Models
# ----------------------------------
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
model_linear = svm.SVC(kernel='linear')
model_rbf = svm.SVC(kernel='rbf')

model_linear.fit(X_train, y_train)
model_rbf.fit(X_train, y_train)

# ----------------------------------
# Plot
# ----------------------------------
plt.figure(figsize=(10,4))

# -------- Linear Kernel --------
plt.subplot(1,2,1)

plt.scatter(X[:,0], X[:,1], c=y)
ax = plt.gca()

xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 200)
yy = np.linspace(ylim[0], ylim[1], 200)
XX, YY = np.meshgrid(xx, yy)

grid = np.c_[XX.ravel(), YY.ravel()]
Z = model_linear.predict(grid)
Z = Z.reshape(XX.shape)

plt.contourf(XX, YY, Z, alpha=0.3)
plt.title("Linear Kernel")

# -------- RBF Kernel --------
plt.subplot(1,2,2)

plt.scatter(X[:,0], X[:,1], c=y)
ax = plt.gca()

xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 200)
yy = np.linspace(ylim[0], ylim[1], 200)
XX, YY = np.meshgrid(xx, yy)

grid = np.c_[XX.ravel(), YY.ravel()]
Z = model_rbf.predict(grid)
Z = Z.reshape(XX.shape)

plt.contourf(XX, YY, Z, alpha=0.3)
plt.title("RBF Kernel")

plt.tight_layout()
plt.show()