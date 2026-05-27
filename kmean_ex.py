import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix,accuracy_score

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

print(df.head())
X = df.iloc[:, [3, 4]].values
# 📈 Elbow Method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

# K-Means Clustering (k=5)
kmeans = KMeans(n_clusters=5,random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 📊 Cluster Visualization
plt.subplot(1,2,2)

for i in range(5):
    plt.scatter(X[y_kmeans == i, 0],X[y_kmeans == i, 1],label=f'Cluster {i+1}'  )

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0],kmeans.cluster_centers_[:, 1],s=200,marker='X',label='Centroids')
plt.title("Customer Clusters")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.tight_layout()
plt.show()
