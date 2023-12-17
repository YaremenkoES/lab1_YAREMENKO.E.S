import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


n_samples = 300
X, _ = make_blobs(n_samples=n_samples, centers=4, cluster_std=1.0, random_state=42)


def moving_average(X, n):
    cumsum = np.cumsum(np.insert(X, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)

scores = []
for n_clusters in range(2, 16):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(X)
    labels = kmeans.labels_
    scores.append(silhouette_score(X, labels))

optimal_clusters = np.argmax(scores) + 2  # Adding 2 to account for the range


kmeans_optimal = KMeans(n_clusters=optimal_clusters)
kmeans_optimal.fit(X)
cluster_centers = kmeans_optimal.cluster_centers_


plt.figure(figsize=(15, 5))

# Initial points
plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("Initial Points")

# Cluster centers (moving average method)
plt.subplot(132)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200)
plt.title("Cluster Centers (Moving Average Method)")

# Bar chart of scores for different numbers of clusters
plt.subplot(133)
plt.bar(range(2, 16), scores)
plt.title("Silhouette Score vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")

# Plot clustered data with clustering areas
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=kmeans_optimal.labels_, s=50, cmap='viridis')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200)
plt.title("Clustered Data with Clustering Areas")

plt.show()
