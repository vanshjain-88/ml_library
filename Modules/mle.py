# mle.py
import numpy as np

class MLEstimator:
    """
    A general-purpose MLE estimator that computes the class-wise
    mean vectors and covariance matrices from labeled data.

    Usage:
        mle = MLEstimator()
        mle.fit(X, y)
        means, covariances = mle.get_params()
    """

    def __init__(self):
        self.means = {}
        self.covariances = {}

    def fit(self, X, y):
        """
        Compute MLE estimates: mean and covariance for each class.

        Parameters:
        - X: np.ndarray of shape (n_samples, n_features)
        - y: np.ndarray of shape (n_samples,) with class labels
        """
        X = np.asarray(X)
        y = np.asarray(y)

        unique_classes = np.unique(y)

        for cls in unique_classes:
            X_cls = X[y == cls]
            self.means[cls] = np.mean(X_cls, axis=0)
            self.covariances[cls] = np.cov(X_cls, rowvar=False)

    def get_params(self):
        """
        Returns:
        - means: dict[class_label -> mean vector]
        - covariances: dict[class_label -> covariance matrix]
        """
        return self.means, self.covariances
