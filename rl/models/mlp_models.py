from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd, adam
from keras.regularizers import l2

import numpy as np

class FeedForwardModel(object):
    def __init__(self, in_size, out_sizes, verbose = False):
        self.verbose = verbose
        model = Sequential()
        model.add(Dense(output_dim=out_sizes[0], input_dim=in_size, activation=('relu' if len(out_sizes) > 1 else 'linear'), init='normal', kernel_regularizer=l2(1e-2)))
        for i in range(1, len(out_sizes)):
            model.add(Dense(output_dim=out_sizes[i], activation=('relu' if i < len(out_sizes) - 1 else 'linear'), init='normal', kernel_regularizer=l2(1e-2)))

        print(model.summary())
        model.compile(sgd(lr=.2), "mse")
        # model.compile(adam(lr=1e-1), "mse")
        self._model = model

    def predict(self, s):
        if self.verbose:
            print("predict X: %s" % s)
        s = np.atleast_2d([s])
        # if self.normalize:
        #     s = self.scaler.transform(s)

        # X = self.rbfs.transform(s)
        res = np.squeeze(self._model.predict(s))
        if self.verbose:
            print("predict Y: %s" % res)
        return res

    def update(self, s, a, y):
        s = np.atleast_2d([s])
        y = np.atleast_2d([y])
        if self.verbose:
            print("update s; y: %s; %s" % (s, y))
        # if self.normalize:
        #     s = self.scaler.transform(s)

        # X = self.rbfs.transform(s)
        self._model.train_on_batch(s, y)