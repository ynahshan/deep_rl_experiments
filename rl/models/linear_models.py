# from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler

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
    def __init__(self, in_size, num_features, output_size, gammmas, normalize = True, verbose=False):
        self.in_dim = in_size
        self.out_dim = output_size
        self.normalize = normalize
        self.verbose = verbose
        self.models = []

        samplers = []
        for g in gammmas:
            samplers.append((("rbf_%f" % g), RBFSampler(gamma=g, n_components=num_features)))

        self.rbfs = FeatureUnion(samplers)
        self.scaler = StandardScaler()

        observation_examples = np.zeros(in_size)
        if observation_examples.ndim == 1:
            observation_examples = observation_examples.reshape((observation_examples.shape[0], 1))
        feature_examples = self.rbfs.fit_transform(np.atleast_2d(observation_examples))
        self.dimensions = feature_examples.shape[1]

        for _ in range(output_size):
            # reg = SGDRegressor(learning_rate='constant')
            reg = SGDRegressor(self.dimensions)
            self.models.append(reg)

    def fit_features(self, observations, env):
        if observations.ndim == 1:
            observations = observations.reshape((observations.shape[0], 1))

        if self.normalize:
            self.scaler.fit(observations)
            self.rbfs.fit(self.scaler.transform(np.atleast_2d(observations)))
        else:
            self.rbfs.fit(np.atleast_2d(observations))

        for model in self.models:
            s = env.reset()
            if self.normalize:
                temp = self.scaler.transform(np.atleast_2d(np.array(s)))
            else:
                temp = np.atleast_2d(np.array(s))
            model.partial_fit(self.rbfs.transform(temp), [0])

    def predict(self, s):
        s = np.atleast_2d([s])
        if self.normalize:
            s = self.scaler.transform(s)

        X = self.rbfs.transform(s)
        res = np.array([m.predict(X)[0] for m in self.models])
        if self.verbose:
            print("predict res: %s" % res)
        return res

    def update(self, s, a, y):
        s = np.atleast_2d([s])
        if self.normalize:
            s = self.scaler.transform(s)

        X = self.rbfs.transform(s)
        self.models[a].partial_fit(X, [y[a]])

    # def adjust(self):
    #     for m in self.models:
    #         m.lr /= 10

    def __str__(self):
        return "RBF features model. in_dim: %d, out_dim %d" % (self.in_dim, self.out_dim)