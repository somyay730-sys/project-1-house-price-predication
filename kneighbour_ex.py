import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# -------------------------------
# Load Dataset
# -------------------------------
data = datasets.load_diabetes()
X = data.data
y = data.target

# -------------------------------
# 📊 BEFORE SCALING
# -------------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.histplot(X[:, 0], kde=True)
plt.title("Feature 1 Before Scaling")

plt.subplot(1,2,2)
sns.histplot(X[:, 1], kde=True)
plt.title("Feature 2 Before Scaling")

plt.tight_layout()
plt.show()

# -------------------------------
# FEATURE SCALING
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 📊 AFTER SCALING
# -------------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.histplot(X_scaled[:, 0], kde=True)
plt.title("Feature 1 After Scaling")

plt.subplot(1,2,2)
sns.histplot(X_scaled[:, 1], kde=True)
plt.title("Feature 2 After Scaling")

plt.tight_layout()
plt.show()

# -------------------------------
# TRAIN-TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# KNN REGRESSOR + ACCURACY GRAPH
# -------------------------------
k_values = range(1, 21)
scores = []

for k in k_values:
    model = KNeighborsRegressor(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(r2_score(y_test, y_pred))

# -------------------------------
# 📈 Accuracy vs K
# -------------------------------
plt.figure()

plt.plot(k_values, scores, marker='o')

plt.xlabel("K Value")
plt.ylabel("R² Score")
plt.title("KNN Performance (Diabetes Dataset)")

plt.show()
