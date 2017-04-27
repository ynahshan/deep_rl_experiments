'''
Created on Apr 26, 2017

@author: Yury
'''

import sys
import timeit
import numpy as np

class SarsaAgent(object):
    def __init__(self, eps=1.0, gamma=0.9, alpha=0.1, verbose=False):
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        self.epoch = 0
        self.Q = {}
        self.random_actions = 0
        self.greedy_actions = 0
        self.verbose = verbose
        self.update_counts_sa = {}

    def choose_action(self, env):
        # choose an action based on epsilon-greedy strategy
        r = np.random.rand()
        eps = float(self.eps) / (self.epoch + 1)
        if r < eps:
            # take a random action
            next_move = np.random.choice(len(env.all_actions()))
            self.random_actions += 1
            if self.verbose:
                print("Taking a random action " + env.action_to_str(next_move))
                print("epsilon: %r < %f" % (r, eps))
        else:
            # choose the best action based on current values of states
            # loop through all possible moves, get their values
            # keep track of the best value
            self.greedy_actions += 1
            next_move = self.optimal_action(env)
            if self.verbose:
                print ("Taking a greedy action " + env.action_to_str(next_move))
                
        return next_move

    def take_action(self, env, action):        
        # make the move
        state, r, done, _ = env.step(action)
        
        # if verbose, draw the grid
#         if self.verbose:
#             env.show()
            
        return state, r, done

    def get_action(self, env):
        s = env.state
        if s not in self.Q:
            self.Q[s] = np.zeros(len(env.all_actions()))
        return self.choose_action(env)

    def print_Q(self, Q):
        for s in Q:
            print("%d %s" % (s, str(self.Q[s])))

    def single_episode_train(self, env):
#         start_time = timeit.default_timer()
        # loops until grid is solved
        steps = 0
        # the first (s, r) tuple is the state we start in and 0
        # (since we don't get a reward) for simply starting the game
        # the last (s, r) tuple is the terminal state and the final reward
        # the value for the terminal state is by definition 0, so we don't
        # care about updating it.
        a = self.get_action(env)
        done = False
        while not done:
            s = env.state
            s2, r, done = self.take_action(env, a)

            # we need the next action as well since Q(s,a) depends on Q(s',a')
            # if s2 not in policy then it's a terminal state, all Q are 0
            a2 = self.get_action(env)
            if s not in self.update_counts_sa:
                self.update_counts_sa[s] = np.ones(len(env.all_actions()))
            # we will update Q(s,a) AS we experience the episode
            alpha = self.alpha / self.update_counts_sa[s][a]
            self.update_counts_sa[s][a] += 0.005
            self.Q[s][a] = self.Q[s][a] + alpha * (r + self.gamma * self.Q[s2][a2] - self.Q[s][a])

            a = a2
                
            steps += 1
            # Increase epsilon as workaround to stacking in infinite actions chain
            if steps > env.grid_size * 2 and self.epoch > 1:
                self.epoch /= 2

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
        if verbosity >= 1:
            print("Updating Value function. Policy improvement.")
        
        for i in states:
            if i % 1000 == 0 and verbosity <= 1:
                sys.stdout.write('.')
                sys.stdout.flush()

            # Doing create_environment(s) instead similar to exploring starts method
            env = env_factory.create_environment()
            # V(s) has only value if it's not a terminal state 
            if env != None:
                steps = self.single_episode_train(env)
                    
        print()
        return steps

    '''
    Interface method
    '''
    def optimal_action(self, env):
        s = env.state
        if s in self.Q:
            return np.argmax(self.Q[s])
        else:
            # if we didn't seen this state before just do random action
            return np.random.choice(env.all_actions())
