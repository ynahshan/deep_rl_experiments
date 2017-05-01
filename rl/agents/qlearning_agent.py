'''
Created on Apr 27, 2017

@author: ynahshan
'''

import sys
import timeit
import numpy as np

class QLearningTabularAgent(object):
    def __init__(self, eps=1.0, gamma=0.9, alpha=0.1, env_descriptor = None, verbose=False):
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        self.epoch = 0
        self.Q = {}
        self.random_actions = 0
        self.greedy_actions = 0
        self.verbose = verbose
        self.update_counts_sa = {}
        self.env_descriptor = env_descriptor

    def choose_action(self, env, s):
        # choose an action based on epsilon-greedy strategy
        r = np.random.rand()
        eps = float(self.eps) / (self.epoch + 1)
        if r < eps:
            # take a random action
            next_move = np.random.choice(env.action_space.n)
            self.random_actions += 1
            if self.verbose:
                if self.env_descriptor != None:
                    print("Taking a random action " + self.env_descriptor.action_to_str(next_move))
                print("epsilon: %r < %f" % (r, eps))
        else:
            # choose the best action based on current values of states
            # loop through all possible moves, get their values
            # keep track of the best value
            self.greedy_actions += 1
            next_move = self.optimal_action(s, env.action_space.n)
            if self.verbose:
                if self.env_descriptor != None:
                    print ("Taking a greedy action " + self.env_descriptor.action_to_str(next_move))
                
        return next_move

    def print_Q(self, Q):
        for s in sorted(Q):
            print("%s %s" % (s, str(self.Q[s])))

    def single_episode_train(self, env):
#         start_time = timeit.default_timer()
        # loops until grid is solved
        steps = 0
        # the first (s, r) tuple is the state we start in and 0
        # (since we don't get a reward) for simply starting the game
        # the last (s, r) tuple is the terminal state and the final reward
        # the value for the terminal state is by definition 0, so we don't
        # care about updating it.
        done = False
        s = env.reset()
        while not done:
            if s not in self.Q:
                self.Q[s] = np.zeros(env.action_space.n)
            # epsilon greedy action selection
            a = self.choose_action(env, s)
            s2, r, done, _ = env.step(a)

            if s2 not in self.Q:
                self.Q[s2] = np.zeros(env.action_space.n)

            if s not in self.update_counts_sa:
                self.update_counts_sa[s] = np.ones(env.action_space.n)

            alpha = self.alpha / self.update_counts_sa[s][a]
            self.update_counts_sa[s][a] += 0.005
            self.Q[s][a] = self.Q[s][a] + alpha * (r + self.gamma * self.Q[s2].max() - self.Q[s][a])
                
            steps += 1
            # Increase epsilon as workaround to stacking in infinite actions chain
            if self.env_descriptor != None and steps > self.env_descriptor.episod_limit and self.epoch > 1:
                self.epoch /= 2
                
            s = s2

        if self.verbose:
            print("\nEpisode finished with reward %f" % r)
            print("Q table:")
            self.print_Q(self.Q)
            print()
        self.epoch += 1
            
#         elapsed = timeit.default_timer() - start_time
#         if verbosity >= 2:
#             print("Solved in %d steps" % len(self.state_history))
#             print("Time to solve grid %.3f[ms]" % (elapsed * 1000))
#             print("Random actions %d, greedy actions %d" % (self.random_actions, self.greedy_actions))

        return steps
            
    def display_functions(self, env):
        policy = {}
        V = {}
        for s in self.Q:
            policy[s] = np.argmax(self.Q[s])
            V[s] = np.max(self.Q[s])
        env.show_values(V)
        env.show_policy(policy)
        pass
    
    def load_model(self, file_name):
#         model = np.fromfile(file_name)
#         self.V = model.reshape(2, int(model.size / 2))[0]
#         self.policy = model.reshape(2, int(model.size / 2))[1]
        pass

    def adjust(self):
        pass

    '''
    Interface method
    '''    
    def save_model(self, file_name):
#         model = np.concatenate((self.V.reshape(1, self.V.size), self.policy.reshape(1, self.policy.size)))
#         model.tofile(file_name)
        pass
    
    '''
    Interface method
    '''
    def single_iteration_train(self, env_factory, states, verbosity=0):
        steps = 0
        for i in states:
            if i % 1000 == 0 and verbosity <= 1:
                sys.stdout.write('.')
                sys.stdout.flush()

            # Doing create_environment(s) instead similar to exploring starts method
            env = env_factory.create_environment()
            # V(s) has only value if it's not a terminal state 
            if env != None:
                if self.verbose:
                    print("Epoch %d." % i)
                steps += self.single_episode_train(env)
                    
        print()
        return steps

    '''
    Interface method
    '''
    def optimal_action(self, s, action_space):
        if s in self.Q:
            return np.argmax(self.Q[s])
        else:
            # if we didn't seen this state before just return rundom_action
            return np.random.choice(action_space)

# from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion
from sklearn.kernel_approximation import RBFSampler

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

        observation_examples = np.array([env.observation_space.sample() for x in range(1000)])
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

class QLearningRbfRegressorAgent(object):
    def __init__(self, env, eps=1.0, gamma=0.9, alpha=0.1, env_descriptor = None, verbose=False):
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        self.epoch = 0
        self.random_actions = 0
        self.greedy_actions = 0
        self.verbose = verbose
        self.update_counts_sa = {}
        self.env_descriptor = env_descriptor
        self.model = RbfRegressor(env.action_space.n, env)

    def adjust(self):
        for m in self.model.models:
            m.lr /= 2

    def choose_action(self, env, s):
        # choose an action based on epsilon-greedy strategy
        r = np.random.rand()
        eps = float(self.eps) / (self.epoch + 1)
        if r < eps:
            # take a random action
            next_move = env.action_space.sample()
            self.random_actions += 1
            if self.verbose:
                if self.env_descriptor != None:
                    print("Taking a random action " + self.env_descriptor.action_to_str(next_move))
                print("epsilon: %r < %f" % (r, eps))
        else:
            # choose the best action based on current values of states
            # loop through all possible moves, get their values
            # keep track of the best value
            self.greedy_actions += 1
            next_move = np.argmax(self.model.predict(s))
            if self.verbose:
                if self.env_descriptor != None:
                    print("Taking a greedy action " + self.env_descriptor.action_to_str(next_move))

        return next_move

    def single_episode_train(self, env):
        steps = 0
        done = False
        s = env.reset()
        while not done:
            # epsilon greedy action selection
            a = self.choose_action(env, s)
            s2, r, done, _ = env.step(a)

            # if s not in self.update_counts_sa:
            #     self.update_counts_sa[s] = np.ones(env.action_space.n)

            # alpha = self.alpha / self.update_counts_sa[s][a]
            # self.update_counts_sa[s][a] += 0.005
            next = self.model.predict(s2)
            G = r + self.gamma * np.max(next)
            self.model.update(s, a, G)

            steps += 1
            # Increase epsilon as workaround to stacking in infinite actions chain
            if self.env_descriptor != None and steps > self.env_descriptor.episod_limit and self.epoch > 1:
                self.epoch /= 2

            s = s2

        # if self.verbose:
        #     print("\nEpisode finished with reward %f" % r)
        #     print("Q table:")
        #     self.print_Q(self.Q)
        #     print()
        self.epoch += 1
        return steps

    '''
    Interface method
    '''
    def optimal_action(self, s, action_space):
        return np.argmax(self.model.predict(s))
