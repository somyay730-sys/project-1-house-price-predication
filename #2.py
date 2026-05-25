import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv('titanic.csv')
df.columns = df.columns.str.strip()

print("First 5 rows:\n", df.head())
print("\nInfo:\n")
df.info()

# -------------------------------
# Handling Missing Values
# -------------------------------

# Age → fill with mean
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Embarked → fill with mode (most frequent)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Cabin → too many missing → drop column
df.drop(columns=['Cabin'], inplace=True)

# -------------------------------
# Encoding Categorical Data
# -------------------------------

le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# -------------------------------
# Feature Scaling
# -------------------------------

scaler = StandardScaler()
df['Fare'] = scaler.fit_transform(df[['Fare']])

# -------------------------------
# Basic Statistics
# -------------------------------

print("\nMean of columns:\n", df.mean(numeric_only=True))
print("\nVariance of columns:\n", df.var(numeric_only=True))

# -------------------------------
# Correlation Matrix
# -------------------------------

corr = df.corr(numeric_only=True)
print("\nCorrelation Matrix:\n", corr)

# -------------------------------
# Visualization
# -------------------------------

# Line plot for Fare
plt.figure()
plt.plot(df['Fare'])
plt.title("Fare Distribution (Scaled)")
plt.xlabel("Index")
plt.ylabel("Fare")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()