import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("Advertising.csv")

# Remove index column if present
df = df.iloc[:, 1:]

print(df.head())

# -------------------------------
# 📊 Feature Relationships
# -------------------------------
sns.pairplot(df)
plt.suptitle("Feature Relationships", y=1.02)
plt.show()

# -------------------------------
# Select one feature for plotting
# -------------------------------
X = df[['TV']].values   # using TV for visualization
y = df['Sales'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Linear Regression
# -------------------------------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

y_pred_lin = lin_model.predict(X_test)

# -------------------------------
# Polynomial Regression (degree=2)
# -------------------------------
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_poly_train, y_train)

y_pred_poly = poly_model.predict(X_poly_test)

# -------------------------------
# 📊 Visualization
# -------------------------------
plt.figure(figsize=(12,5))

# Linear plot
plt.subplot(1,2,1)
plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred_lin)
plt.title("Linear Regression")
plt.xlabel("TV Advertising")
plt.ylabel("Sales")

# Polynomial plot
plt.subplot(1,2,2)
plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred_poly)
plt.title("Polynomial Regression")
plt.xlabel("TV Advertising")
plt.ylabel("Sales")

plt.tight_layout()
plt.show()

# -------------------------------
# Compare Errors
# -------------------------------
print("Linear MSE:", mean_squared_error(y_test, y_pred_lin))
print("Polynomial MSE:", mean_squared_error(y_test, y_pred_poly))