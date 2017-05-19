import sys
import timeit
import numpy as np
import tensorflow as tf

from rl_gym.models.tf_layers import HiddenLayer

class DQNModel:
    def __init__(self, D, K, hidden_layer_sizes, gamma, max_experiences=10000, min_experiences=100, batch_sz=32):
        self.K = K

        # create the graph
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2

        # final layer
        layer = HiddenLayer(M1, K, lambda x: x)
        self.layers.append(layer)

        # collect params for copy
        self.params = []
        for layer in self.layers:
            self.params += layer.params

        # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

        # calculate output and cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = Z
        self.predict_op = Y_hat

        selected_action_values = tf.reduce_sum(
          Y_hat * tf.one_hot(self.actions, K),
          reduction_indices=[1]
        )

        cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
        self.train_op = tf.train.AdamOptimizer(10e-3).minimize(cost)
        # self.train_op = tf.train.AdagradOptimizer(10e-3).minimize(cost)
        # self.train_op = tf.train.MomentumOptimizer(10e-4, momentum=0.9).minimize(cost)
        # self.train_op = tf.train.GradientDescentOptimizer(10e-5).minimize(cost)

        # create replay memory
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_sz = batch_sz
        self.gamma = gamma

    def set_session(self, session):
        self.session = session

    def copy_from(self, other):
        # collect all the ops
        ops = []
        my_params = self.params
        other_params = other.params
        for p, q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        # now run them all
        self.session.run(ops)

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def train(self, target_network):
        # sample a random batch from buffer, do an iteration of GD
        if len(self.experience['s']) < self.min_experiences:
            # don't do anything if we don't have enough experience
            return

        # randomly select a batch
        idx = np.random.choice(len(self.experience['s']), size=self.batch_sz, replace=False)
        # print("idx:", idx)
        states = [self.experience['s'][i] for i in idx]
        actions = [self.experience['a'][i] for i in idx]
        rewards = [self.experience['r'][i] for i in idx]
        next_states = [self.experience['s2'][i] for i in idx]
        dones = [self.experience['done'][i] for i in idx]
        next_Q = np.max(target_network.predict(next_states), axis=1)
        targets = [r + self.gamma*next_q if not done else r for r, next_q, done in zip(rewards, next_Q, dones)]

        # call optimizer
        self.session.run(
          self.train_op,
          feed_dict={
            self.X: states,
            self.G: targets,
            self.actions: actions
          }
        )

    def add_experience(self, s, a, r, s2, done):
        if len(self.experience['s']) >= self.max_experiences:
            self.experience['s'].pop(0)
            self.experience['a'].pop(0)
            self.experience['r'].pop(0)
            self.experience['s2'].pop(0)
            self.experience['done'].pop(0)
        self.experience['s'].append(s)
        self.experience['a'].append(a)
        self.experience['r'].append(r)
        self.experience['s2'].append(s2)
        self.experience['done'].append(done)

class DQNAgent(object):
    def __init__(self, model, target_model, eps=1.0, eps_decay = 0.99, eps_min=0, gamma=0.9, copy_period = 50, verbose=False):
        self.model = model
        self.target_model = target_model
        self.eps = eps
        self.gamma = gamma
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.copy_period = copy_period
        self.random_actions = 0
        self.greedy_actions = 0
        self.epoch = 0
        self.verbose = verbose

        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)
        model.set_session(self.session)
        target_model.set_session(self.session)

    def choose_action(self, env, s, y_s):
        # choose an action based on epsilon-greedy strategy
        r = np.random.rand()
        if r < self.eps:
            # take a random action
            next_move = env.action_space.sample()
            self.random_actions += 1
            if self.verbose:
                print("Taking a random action %s" % next_move)
                print("epsilon: %r < %f" % (r, self.eps))
        else:
            self.greedy_actions += 1
            next_move = np.argmax(y_s)
            if self.verbose:
                print("Taking a greedy action %s" % next_move)

        return next_move

    '''
    Interface method
    '''
    def single_episode_train(self, env):
        steps = 0
        done = False
        s = env.reset()
        total_return = 0
        actions = []
        n_actions = env.action_space.n
        while not done:
            # predict Q value for current state s
            y_s = self.model.predict(s)
            if self.verbose:
                print("Step %d:" % steps)
                print("Observed state %s. Predicted value: %s" % (s, y_s))
            # epsilon greedy action selection
            a = self.choose_action(env, s, y_s)
            actions.append(a)
            s2, r, done, _ = env.step(a)
            total_return += r

            # update the model
            self.model.add_experience(s, a, r, s2, done)
            self.model.train(self.target_model)

            steps += 1
            s = s2

            if steps % self.copy_period == 0:
                self.target_model.copy_from(self.model)

        self.epoch += 1
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
        if self.verbose:
            print("Actions sequence for this episode:")
            print(actions)

        self.target_model.copy_from(self.model)

        return steps, total_return, r

    '''
    Interface method
    '''
    def optimal_action(self, s, action_space):
        y = self.target_model.predict(s).squeeze()
        return np.argmax(y)