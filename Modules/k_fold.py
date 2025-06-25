import numpy as np
from sklearn.metrics import mean_squared_error  # You can replace with your own if needed


class KFoldCrossValidator:
    """
    Performs k-Fold Cross-Validation for any model with fit(X, y) and predict(X).
    """

    def __init__(self, model, k=5, metric=mean_squared_error):
        """
        model: any object with fit() and predict()
        k: number of folds
        metric: function to evaluate performance (default: MSE)
        """
        self.model = model
        self.k = k
        self.metric = metric

    def evaluate(self, X, y):
        """
        Performs k-fold CV and returns average score (lower is better).
        """
        n_samples = X.shape[0]
        indices = np.random.permutation(n_samples)
        fold_size = n_samples // self.k
        scores = []

        for i in range(self.k):
            val_idx = indices[i * fold_size:(i + 1) * fold_size]
            train_idx = np.setdiff1d(indices, val_idx)

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            model_instance = self._clone_model()
            model_instance.fit(X_train, y_train)
            y_pred = model_instance.predict(X_val)

            score = self.metric(y_val, y_pred)
            scores.append(score)

        return np.mean(scores)

    def _clone_model(self):
        return type(self.model)()  # Re-instantiates a fresh model
