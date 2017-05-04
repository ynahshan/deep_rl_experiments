from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd, adam, rmsprop, adagrad, adadelta
from keras.regularizers import l2, l1, l1_l2

from sklearn.preprocessing import StandardScaler

import numpy as np

class FeedForwardModel(object):
    def __init__(self, in_size, out_sizes, normalize = True, verbose = False):
        self.verbose = verbose
        self.normalize = normalize
        self.regularizer = 10e-5
        self.scaler = StandardScaler()
        model = Sequential()
        model.add(Dense(units=out_sizes[0], input_dim=in_size, activation=('relu' if len(out_sizes) > 1 else 'sigmoid'), kernel_initializer='normal', kernel_regularizer=l2(self.regularizer), bias_regularizer=l2(self.regularizer)))
        # model.add(Dense(units=out_sizes[0], input_dim=in_size, activation=('relu' if len(out_sizes) > 1 else 'linear')))
        for i in range(1, len(out_sizes)):
            model.add(Dense(units=out_sizes[i], activation=('relu' if i < len(out_sizes) - 1 else 'linear'), kernel_initializer='normal', kernel_regularizer=l2(self.regularizer), bias_regularizer=l2(self.regularizer)))
            # model.add(Dense(units=out_sizes[i], activation=('relu' if i < len(out_sizes) - 1 else 'linear')))

        print(model.summary())
        model.compile(sgd(lr=10e-2), "mse")
        # model.compile(adam(), "mse")
        # model.compile(adadelta(), "mse")
        self._model = model

    def fit_features(self, observations, env):
        if observations.ndim == 1:
            observations = observations.reshape((observations.shape[0], 1))

        if self.normalize:
            self.scaler.fit(observations)

    def predict(self, s):
        if self.verbose:
            print("predict X: %s" % s)
        s = np.atleast_2d([s])
        if self.normalize:
            s = self.scaler.transform(s)

        res = np.squeeze(self._model.predict(s))
        if self.verbose:
            print("predict Y: %s" % res)
        return res

    def update(self, s, a, y):
        s = np.atleast_2d([s])
        y = np.atleast_2d([y])
        if self.normalize:
            s = self.scaler.transform(s)

        if self.verbose:
            print("update s; y: %s; %s" % (s, y))

        loss = self._model.train_on_batch(s, y)
        if self.verbose:
            print("Loss: %f" % loss)
        # for layer in self._model.layers:
        #     weights = np.array(layer.get_weights())
        #     print(weights[0].mean())
        #     print(weights[1].mean())
        #     print()