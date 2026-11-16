"""
Ridge regression on random Fourier features, following Kelly, Malamud and Zhou (2024),
"The Virtue of Complexity in Return Prediction".

The inputs are expanded into P random features sin(gamma * W x) and cos(gamma * W x), with
W drawn from a standard normal, and next-day returns are regressed on them with a ridge penalty.
Forecasts are averaged over several random draws of W, as in the paper.
"""
import numpy as np
from sklearn.linear_model import Ridge


class RFFRidge:
    def __init__(self, n_features=1000, gamma=2.0, alpha=1000.0, seeds=10):
        self.P, self.gamma, self.alpha, self.seeds = n_features, gamma, alpha, seeds

    def _expand(self, X, W):
        Z = self.gamma * X @ W
        return np.hstack([np.sin(Z), np.cos(Z)])

    def fit(self, X, r):
        X = np.asarray(X, float)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-12
        Xs = (X - self.mu_) / self.sd_
        self.models_ = []
        for s in range(self.seeds):
            W = np.random.default_rng(s).standard_normal((X.shape[1], self.P // 2))
            self.models_.append((W, Ridge(alpha=self.alpha).fit(self._expand(Xs, W), r)))
        return self

    def predict(self, X):
        Xs = (np.asarray(X, float) - self.mu_) / self.sd_
        return np.mean([m.predict(self._expand(Xs, W)) for W, m in self.models_], axis=0)
