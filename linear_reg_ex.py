
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=pd.read_csv('titanic.csv')

print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())

# 3. Handle Missing Values
print("\nMissing Values Before:\n", df.isnull().sum())

# Fill numerical column
df['Age'] = df['Age'].fillna(df['Age'].median())
# Fill categorical column
df['Embarked'].fillna(df['Embarked'].mode()[0])

print("\nMissing Values After:\n", df.isnull().sum())

# 4. Encode Categorical Variables
le = LabelEncoder()

df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])
df['Pclass'] = le.fit_transform(df['Pclass'])

# 5. Feature Scaling
scaler = StandardScaler()

df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

# 6. Basic Visualizations
# Survival Count
plt.subplot(3,1,1)
sns.countplot(x='Survived', data=df)
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")


# Age Distribution
plt.subplot(3,1,2)
sns.histplot(df['Age'], kde=True)
plt.title("Age Distribution")


# Fare vs Age (with survival)
plt.subplot(3,1,3)
sns.scatterplot(x='Fare', y='Age', hue='Survived', data=df)
plt.title("Fare vs Age (Survival Highlighted)")
plt.tight_layout()
plt.show()

# 7. Mean and Variance
print("\nMean:\n", df.mean(numeric_only=True))
print("\nVariance:\n", df.var(numeric_only=True))

# 8. Correlation Matrix
corr = df.corr(numeric_only=True)
print("\nCorrelation Matrix:\n", corr)

# 9. Heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 10. Scatter Plot Matrix (Pairplot)
sns.pairplot(df[['Age', 'Fare', 'Survived']])
plt.show()

x = df[['Age', 'Fare', 'Survived']]
y = df['Survived']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
logreg = LinearRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
print(logreg.score(x_test, y_test))

plt.plot(y_test, y_pred)
plt.show()
