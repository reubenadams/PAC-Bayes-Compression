import torch
from sklearn.cluster import kmeans_plusplus
from sklearn.metrics import pairwise_distances_argmin

def kmeans_init_with_labels(X, n_clusters, random_state=None):
    # Get initial centers using k-means++ initialization
    centers, indices = kmeans_plusplus(X, n_clusters=n_clusters, random_state=random_state)
    
    # Assign each point to nearest center (this is much faster than doing an iteration)
    labels = pairwise_distances_argmin(X, centers)
    
    return centers, labels

# Usage:
X = torch.randn(200000, 1).numpy()
k = 10000
centers, labels = kmeans_init_with_labels(X, n_clusters=k, random_state=0)