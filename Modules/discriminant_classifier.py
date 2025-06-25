import numpy as np

class LDAClassifier:
    """
    Linear Discriminant Analysis Classifier.
    Assumes a shared covariance matrix across all classes.
    """

    def __init__(self):
        self.means = None
        self.cov = None
        self.cov_inv = None
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_features = X.shape[1]
        self.means = []
        cov_sum = np.zeros((n_features, n_features))
        total_samples = 0

        for cls in self.classes:
            X_cls = X[y == cls]
            mean_cls = np.mean(X_cls, axis=0)
            self.means.append(mean_cls)
            cov_sum += np.cov(X_cls, rowvar=False, bias=True) * X_cls.shape[0]
            total_samples += X_cls.shape[0]

        self.cov = cov_sum / total_samples
        self.cov_inv = np.linalg.inv(self.cov)
        self.means = np.array(self.means)

    def predict(self, X):
        scores = []
        for mean in self.means:
            score = (mean @ self.cov_inv @ X.T).T - 0.5 * (mean @ self.cov_inv @ mean)
            scores.append(score)
        return np.argmax(scores, axis=0)


class QDAClassifier:
    """
    Quadratic Discriminant Analysis Classifier.
    Uses class-specific means and covariances.
    """

    def __init__(self):
        self.means = None
        self.covs = None
        self.cov_invs = []
        self.cov_dets = []
        self.classes = None

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.means = []
        self.covs = []

        for cls in self.classes:
            X_cls = X[y == cls]
            mean_cls = np.mean(X_cls, axis=0)
            cov_cls = np.cov(X_cls, rowvar=False, bias=True)

            self.means.append(mean_cls)
            self.covs.append(cov_cls)
            self.cov_invs.append(np.linalg.inv(cov_cls))
            self.cov_dets.append(np.linalg.det(cov_cls))

        self.means = np.array(self.means)
        self.covs = np.array(self.covs)

    def predict(self, X):
        scores = []
        for mean, cov_inv, log_det in zip(self.means, self.cov_invs, np.log(self.cov_dets)):
            diff = X - mean
            score = -0.5 * log_det - 0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            scores.append(score)
        return np.argmax(scores, axis=0)
