import pandas as pd
from imblearn.base import SamplerMixin
from sklearn.ensemble import IsolationForest
from imblearn.combine import SMOTEENN


class OutlierRemover(SamplerMixin):
    def __init__(self, seed=666, activate=True):
        self.seed = seed
        self.activate = activate
        self.detector = IsolationForest(n_jobs=-1, random_state=seed)

    def fit(self, X, y):
        if self.activate:
            self.detector.fit(X)
        # Return fit object
        return self

    def _sample(self, X, y):
        if self.activate:
            y_pred = self.detector.predict(X)
            sampled_X = X[y_pred == 1]
            sampled_y = y[y_pred == 1]
            return sampled_X, sampled_y
        else:
            return X, y

    def sample(self, X, y):
        return self._sample(X, y)


class OUSampler(SamplerMixin):
    def __init__(self, seed=666, activate=True):
        self.seed = seed
        self.activate = activate
        self.sampler = SMOTEENN(random_state=self.seed, n_jobs=-1)

    def fit(self, X, y):
        if self.activate:
            self.sampler.fit(X, y)
        # Return fit object
        return self

    def _sample(self, X, y):
        if self.activate:
            sampled_X, sampled_y = self.sampler.sample(X, y)
            sampled_X = pd.DataFrame(sampled_X, columns=X.columns)
            sampled_y = pd.Series(sampled_y, name=y.name)
            return sampled_X, sampled_y
        else:
            return X, y

    def sample(self, X, y):
        return self._sample(X, y)
