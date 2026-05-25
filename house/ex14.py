import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Dataset
df = pd.read_csv("house-prices.csv")

print("--- Dataset Head ---")
print(df.head())

# -------------------------------
# 🛠️ HANDLE MISSING DATA (Fixed the inplace=True bug)
# -------------------------------
df['Bedrooms'] = df['Bedrooms'].fillna(df['Bedrooms'].mean())
df['Bathrooms'] = df['Bathrooms'].fillna(df['Bathrooms'].mean())

# Convert categorical variable (Just in case you want to use 'Brick' later!)
df['Brick'] = df['Brick'].map({'Yes': 1, 'No': 0})

# -------------------------------
# 🤖 LINEAR REGRESSION
# -------------------------------
X = df[['Bedrooms', 'Bathrooms']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print("MSE:", mean_squared_error(y_test, y_pred))

# 📊 VISUALIZE REGRESSION (Fixed 2D Plotting)
plt.figure(figsize=(8, 6))

# Plot actual vs predicted
plt.scatter(y_test, y_pred, alpha=0.7, color='b')

# Plot the perfect prediction line (diagonal)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted House Prices")
plt.grid(True)
plt.show()