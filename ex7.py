import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# -------------------------------
# Load Dataset
# -------------------------------
df = pd.read_csv("titanic.csv")

print(df.head())

# -------------------------------
# 📊 BEFORE PREPROCESSING
# -------------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution (Before)")

plt.subplot(1,2,2)
sns.countplot(x='Sex', data=df)
plt.title("Gender Distribution (Before)")

plt.tight_layout()
plt.show()

# -------------------------------
# PREPROCESSING
# -------------------------------

# Fill missing Age with mean
df['Age'].fillna(df['Age'].mean(), inplace=True)

# Fill missing Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop unnecessary columns
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Encode categorical variables
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# -------------------------------
# 📊 AFTER PREPROCESSING
# -------------------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution (After)")

plt.subplot(1,2,2)
sns.countplot(x='Sex', data=df)
plt.title("Gender (Encoded)")

plt.tight_layout()
plt.show()

# -------------------------------
# TRAIN MODEL
# -------------------------------
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# -------------------------------
# 📊 CONFUSION MATRIX HEATMAP
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.title("Logistic Regression Performance")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()