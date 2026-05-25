import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

# -------------------------------
# LOAD DATA (FIXED)
# -------------------------------
df = pd.read_csv("spam.csv", sep='\t', encoding='latin-1')

df = df.iloc[:, :2]
df.columns = ['label', 'text']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

print(df.head())
print(df.shape)

# -------------------------------
# CLASS DISTRIBUTION
# -------------------------------
plt.figure(figsize=(5,4))
sns.countplot(x='label', data=df)
plt.title("Class Distribution (0 = Ham, 1 = Spam)")
plt.show()

# -------------------------------
# VECTORIZATION
# -------------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# NAIVE BAYES
# -------------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# -------------------------------
# CONFUSION MATRIX
# -------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()