# pca.py
import numpy as np

class PCA:
    """
    Principal Component Analysis from scratch using NumPy.

    Usage:
        pca = PCA(variance_threshold=0.95)
        pca.fit(X)
        X_reduced = pca.transform(X)
    """

    def __init__(self, variance_threshold=0.95):
        self.variance_threshold = variance_threshold
        self.mean = None
        self.components = None
        self.num_components = None

    def fit(self, X):
        X = np.asarray(X)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        N = X.shape[0]
        cov = np.dot(X_centered.T, X_centered) / (N - 1)

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Choose number of components
        cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        self.num_components = np.argmax(cumulative_variance >= self.variance_threshold) + 1

        self.components = eigenvectors[:, :self.num_components]

    def transform(self, X):
        if self.mean is None or self.components is None:
            raise ValueError("You must call fit() before transform().")
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
