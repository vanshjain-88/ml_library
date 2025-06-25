import numpy as np

class DecisionStump:
    def __init__(self, n_cuts=3):
        self.n_cuts = n_cuts
        self.j = None
        self.th = None
        self.pol = None

    def fit(self, X, y, sample_weights):
        n, d = X.shape
        best_err = np.inf
        for j in range(d):
            col = X[:, j]
            cuts = np.linspace(col.min(), col.max(), self.n_cuts + 2)[1:-1]
            for th in cuts:
                for pol in [+1, -1]:
                    pred = np.where(col < th, pol, -pol)
                    err = np.sum(sample_weights * (pred != y))
                    if err < best_err:
                        best_err = err
                        self.j = j
                        self.th = th
                        self.pol = pol
        return best_err

    def predict(self, X):
        return np.where(X[:, self.j] < self.th, self.pol, -self.pol)


class AdaBoost:
    def __init__(self, T=100, n_cuts=3):
        self.T = T
        self.n_cuts = n_cuts
        self.stumps = []
        self.betas = []

    def fit(self, X, y):
        n = len(y)
        w = np.full(n, 1 / n)
        F = np.zeros(n)

        for t in range(self.T):
            stump = DecisionStump(n_cuts=self.n_cuts)
            err = stump.fit(X, y, w)
            err = max(err, 1e-16)  # avoid division by 0
            beta = 0.5 * np.log((1 - err) / err)

            pred = stump.predict(X)
            w *= np.exp(-beta * y * pred)
            w /= w.sum()

            self.stumps.append(stump)
            self.betas.append(beta)

    def predict(self, X):
        F = np.zeros(X.shape[0])
        for stump, beta in zip(self.stumps, self.betas):
            F += beta * stump.predict(X)
        return np.sign(F)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
