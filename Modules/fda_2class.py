# fda_2class.py
import numpy as np

class FDA2Class:
    """
    Fisher's Discriminant Analysis (2-Class closed-form version).

    Computes the optimal projection vector using:
        w = SW^-1 (mu1 - mu2)
    
    Usage:
        fda = FDA2Class()
        fda.fit(X, y)
        X_proj = fda.transform(X)
    """

    def __init__(self):
        self.w = None
        self.mu1 = None
        self.mu2 = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        class_labels = np.unique(y)

        if len(class_labels) != 2:
            raise ValueError("FDA2Class only supports two classes.")

        # Split data
        X1 = X[y == class_labels[0]]
        X2 = X[y == class_labels[1]]

        self.mu1 = np.mean(X1, axis=0)
        self.mu2 = np.mean(X2, axis=0)

        # Within-class scatter matrix
        S1 = sum([(x - self.mu1).reshape(-1, 1) @ (x - self.mu1).reshape(1, -1) for x in X1])
        S2 = sum([(x - self.mu2).reshape(-1, 1) @ (x - self.mu2).reshape(1, -1) for x in X2])
        SW = S1 + S2

        # Optimal projection vector
        self.w = np.linalg.inv(SW) @ (self.mu1 - self.mu2).reshape(-1, 1)

    def transform(self, X):
        if self.w is None:
            raise ValueError("Call fit() before transform().")
        return X @ self.w

    def get_projection_vector(self):
        return self.w
