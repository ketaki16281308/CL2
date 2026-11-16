#Implement Agglomerative hierarchical clustering algorithm using appropriate dataset
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

iris = load_iris()
X = iris.data

model = AgglomerativeClustering(n_clusters=3, linkage='ward')
y = model.fit_predict(X)

print("Cluster Labels:")
print(y)

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Agglomerative Hierarchical Clustering")
plt.show()

Z = linkage(X, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(Z, truncate_mode='lastp', p=20)
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
