import numpy as np

np.random.seed(0)

import tensorflow

tensorflow.set_random_seed(0)

from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd, adam, rmsprop, adagrad, adadelta
from keras.regularizers import l2, l1, l1_l2

import matplotlib.pyplot as plt


def small_net(mid_size, non_linear = 'tanh', initializer='lecun_uniform'):
    model = Sequential()
    model.add(Dense(units=mid_size, input_dim=2, activation=non_linear, kernel_initializer=initializer))
    model.add(Dense(units=int(mid_size), activation=non_linear, kernel_initializer=initializer))
    model.add(Dense(units=int(mid_size), activation=non_linear, kernel_initializer=initializer))
    model.add(Dense(units=int(mid_size / 2), activation=non_linear, kernel_initializer=initializer))
    model.add(Dense(units=4, activation='linear', kernel_initializer=initializer))
    model.compile(adagrad(), "mse")
    print(model.summary())
    return model


if __name__ == '__main__':
    model = small_net(mid_size=64, non_linear='tanh', initializer='lecun_uniform')
    x = np.array([[0.1, -0.5]])
    y = np.array([[1.2, 1.8, -2.0, 1.3]])

    print("X: %s, Y: %s" % (x, y))

    loss = []
    for i in range(3):
        print("Predication: %s" % model.predict(x))
        loss.append(model.train_on_batch(x, y))
        print("Update loss: %s" % loss[-1])

    plt.plot(loss)
    plt.show()
