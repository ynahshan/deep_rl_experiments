# from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.kernel_approximation import RBFSampler

import numpy as np

class SGDRegressor:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 10e-2

    def partial_fit(self, X, Y):
        self.w += self.lr*(Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)

class RbfRegressor(object):
    def __init__(self, output_size, env):
        self.models = []
        n_cmp = 10

        self.rbfs = FeatureUnion([
            # ("rbf1", RBFSampler(gamma=0.05, n_components=n_cmp)),
            ("rbf2", RBFSampler(gamma=0.1, n_components=n_cmp)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=n_cmp)),
            ("rbf4", RBFSampler(gamma=1.0, n_components=n_cmp)),
            ("rbf5", RBFSampler(gamma=1.5, n_components=n_cmp)),
            # ("rbf6", RBFSampler(gamma=2.0, n_components=n_cmp))
        ])

        observation_examples = np.array([env.observation_space.sample() for x in range(100)])
        if observation_examples.ndim == 1:
            observation_examples = observation_examples.reshape((observation_examples.shape[0], 1))
        feature_examples = self.rbfs.fit_transform(np.atleast_2d(observation_examples))
        self.dimensions = feature_examples.shape[1]
        for _ in range(output_size):
            reg = SGDRegressor(self.dimensions)
            self.models.append(reg)

    def predict(self, s):
        temp = np.atleast_2d(np.array(s))
        X = self.rbfs.transform(temp)
        res = np.array([m.predict(X)[0] for m in self.models])
        return res

    def update(self, s, a, G):
        temp = np.atleast_2d(np.array(s))
        X = self.rbfs.transform(temp)
        self.models[a].partial_fit(X, [G])

    def adjust(self):
        for m in self.models:
            m.lr /= 2