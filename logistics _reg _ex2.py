import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# -------------------------------
# Load Dataset
# -------------------------------
wine = datasets.load_wine()
X = wine.data
y = wine.target

# -------------------------------
# 📊 BEFORE NORMALIZATION
# -------------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.histplot(X[:, 0], kde=True)
plt.title("Feature 1 Before Normalization")

plt.subplot(1,2,2)
sns.histplot(X[:, 1], kde=True)
plt.title("Feature 2 Before Normalization")

plt.tight_layout()
plt.show()

# -------------------------------
# NORMALIZATION
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 📊 AFTER NORMALIZATION
# -------------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.histplot(X_scaled[:, 0], kde=True)
plt.title("Feature 1 After Normalization")

plt.subplot(1,2,2)
sns.histplot(X_scaled[:, 1], kde=True)
plt.title("Feature 2 After Normalization")

plt.tight_layout()
plt.show()

# -------------------------------
# TRAIN MODEL
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# -------------------------------
# 📊 CONFUSION MATRIX
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Logistic Regression Output")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()
