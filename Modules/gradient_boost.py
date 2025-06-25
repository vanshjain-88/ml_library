import numpy as np

class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, loss='squared', splits=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.splits = splits if splits is not None else np.linspace(0, 1, 20)
        self.F0 = None
        self.stumps = []

    def _fit_stump(self, x, residuals):
        best_loss = float('inf')
        best_s, best_lm, best_rm = 0, 0, 0
        for s in self.splits:
            mask = x <= s
            if not mask.any() or mask.all():
                continue
            left, right = residuals[mask], residuals[~mask]
            lm, rm = left.mean(), right.mean()
            loss = ((left - lm) ** 2).sum() + ((right - rm) ** 2).sum()
            if loss < best_loss:
                best_loss = loss
                best_s, best_lm, best_rm = s, lm, rm
        return best_s, best_lm, best_rm

    def fit(self, x, y):
        self.F0 = np.mean(y)
        F = np.full_like(y, self.F0)
        self.stumps = []

        for _ in range(self.n_estimators):
            residuals = (y - F) if self.loss == 'squared' else np.sign(y - F)
            s, lm, rm = self._fit_stump(x, residuals)
            self.stumps.append((s, lm, rm))
            F += self.learning_rate * np.where(x <= s, lm, rm)

    def predict(self, x, n_iters=None):
        pred = np.full_like(x, self.F0)
        n_iters = len(self.stumps) if n_iters is None else n_iters
        for i in range(n_iters):
            s, lm, rm = self.stumps[i]
            pred += self.learning_rate * np.where(x <= s, lm, rm)
        return pred
