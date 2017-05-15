import numpy as np
import tensorflow as tf

# class HiddenLayer:
#     def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
#         self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
#         self.use_bias = use_bias
#         if use_bias:
#             self.b = tf.Variable(np.zeros(M2).astype(np.float32))
#         self.f = f
#
#     def forward(self, X):
#         if self.use_bias:
#             a = tf.matmul(X, self.W) + self.b
#         else:
#             a = tf.matmul(X, self.W)
#         return self.f(a)

class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
        self.W = tf.Variable(tf.random_normal(shape=(M1, M2)))
        self.params = [self.W]
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
            self.params.append(self.b)
        self.f = f

    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        return self.f(a)