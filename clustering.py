
import numpy as np
from scipy.spatial.distance import cdist

def k_means(X, k, max_iters=100, tol=1e-6):
    n_samples, n_features = X.shape
    # Randomly initialize centroids
    np.random.seed(0)  # For reproducibility
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    labels = np.zeros(n_samples, dtype=int)
    
    for _ in range(max_iters):
        # Compute distances from each point to each centroid
        dist_matrix = cdist(X, centroids, 'euclidean')
        # Assign each point to the nearest centroid
        new_labels = np.argmin(dist_matrix, axis=1)
        # Update centroids
        new_centroids = np.zeros((k, n_features))
        for i in range(k):
            cluster_points = X[new_labels == i]
            if len(cluster_points) == 0:
                # Handle empty clusters by keeping the previous centroid
                new_centroids[i] = centroids[i]
            else:
                new_centroids[i] = np.mean(cluster_points, axis=0)
        
        # Check for convergence (if centroids don't change within tolerance)
        if np.all(np.linalg.norm(centroids - new_centroids, axis=1) < tol):
            break
        
        centroids = new_centroids
        labels = new_labels
    
    return centroids, labels

# Example usage
if __name__ == "__main__":
    # Example dataset (replace with your own data)
    X = np.array([
        [1, 2], [1, 4], [1, 0],
        [4, 2], [4, 4], [4, 0]
    ])
    k = 2  # Number of clusters
    centroids, labels = k_means(X, k)
    print("Final centroids:\n", centroids)
    print("Cluster labels:\n", labels)