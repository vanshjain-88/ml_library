# fda.py
import numpy as np
from scipy.linalg import eigh

class FDA:
    """
    Fisher's Discriminant Analysis (multiclass version).

    Usage:
        fda = FDA()
        fda.fit(X, y)
        X_projected = fda.transform(X)
    """

    def __init__(self):
        self.W = None  # Projection matrix

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        classes = np.unique(y)
        n_features = X.shape[1]

        overall_mean = np.mean(X, axis=0)

        S_B = np.zeros((n_features, n_features))
        S_W = np.zeros((n_features, n_features))

        for cls in classes:
            X_cls = X[y == cls]
            mu_cls = np.mean(X_cls, axis=0)
            N_cls = X_cls.shape[0]

            # Between-class scatter
            mean_diff = (mu_cls - overall_mean).reshape(n_features, 1)
            S_B += N_cls * (mean_diff @ mean_diff.T)

            # Within-class scatter
            S_W += np.cov(X_cls, rowvar=False, bias=True) * (N_cls - 1)

        eigenvalues, eigenvectors = eigh(S_B, S_W)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.W = eigenvectors[:, sorted_indices]  # Full projection matrix

    def transform(self, X, n_components=None):
        if self.W is None:
            raise ValueError("You must call fit() before transform().")
        if n_components is None:
            return np.dot(X, self.W)
        return np.dot(X, self.W[:, :n_components])
