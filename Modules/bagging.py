import numpy as np
from decision_tree_classifier import DecisionTreeClassifier

class BaggingClassifier:
    def __init__(self, base_estimator=None, n_estimators=10, max_depth=None, min_samples_leaf=1):
        self.base_estimator = base_estimator or DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=min_samples_leaf
        )
        self.n_estimators = n_estimators
        self.estimators = []

    def _bootstrap_sample(self, X, y):
        indices = np.random.choice(len(X), len(X), replace=True)
        return X.iloc[indices], y.iloc[indices]

    def fit(self, X, y):
        self.estimators = []
        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            estimator = DecisionTreeClassifier(
                max_depth=self.base_estimator.max_depth,
                min_samples_leaf=self.base_estimator.min_samples_leaf
            )
            estimator.fit(X_sample, y_sample)
            self.estimators.append(estimator)

    def predict(self, sample):
        votes = [estimator.predict(sample) for estimator in self.estimators]
        return max(set(votes), key=votes.count)
