'''
Created on Apr 23, 2017

@author: Yury
'''

import sys
import timeit
import numpy as np

class MonteCarloTabularAgent(object):
    def __init__(self, eps=1.0, gamma=0.9, env_descriptor = None, verbose=False):
        self.eps = eps
        self.gamma = gamma
        self.epoch = 0
        self.policy = {}
        self.Q = {}
        self.returns = {}
        self.random_actions = 0
        self.greedy_actions = 0
        self.verbose = verbose
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
                print("epsilog: %r < %f" % (r, eps))
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

    def single_episode_exploration(self, env):
#         start_time = timeit.default_timer()
        # loops until grid is solved
        steps = 0
        states_actions_rewards = []
        s = env.reset()
        a = self.choose_action(env, s)
        states_actions_rewards.append((s, a, 0))
        done = False
        while not done:
            s, r, done, _ = env.step(a)
            if done:
                states_actions_rewards.append((s, None, r))
            else:
                a = self.choose_action(env, s)
                states_actions_rewards.append((s, a, r))
                
            steps += 1
            # Increase epsilon as workaround to stacking in infinite actions chain
            if self.env_descriptor != None and steps > self.env_descriptor.episod_limit and self.epoch > 1:
                self.epoch /= 2
                
        if self.verbose:
            print("Episode finished\n")
        self.epoch += 1
            
#         elapsed = timeit.default_timer() - start_time
#         if verbosity >= 2:
#             print("Solved in %d steps" % len(self.state_history))
#             print("Time to solve grid %.3f[ms]" % (elapsed * 1000))
#             print("Random actions %d, greedy actions %d" % (self.random_actions, self.greedy_actions))

        return states_actions_rewards, steps
            
    def display_functions(self, env):
#         env.show_values(self.V)
        env.show_policy(self.policy)
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
    
    def single_episode_train(self, env):
        states_actions_rewards, steps = self.single_episode_exploration(env)
#                 print(states_actions_rewards)
        # calculate the returns by working backwards from the terminal state
        G = 0
        states_actions_returns = []
        first = True
        for s, a, r in reversed(states_actions_rewards):
            # the value of the terminal state is 0 by definition
            # we should ignore the first state we encounter
            # and ignore the last G, which is meaningless since it doesn't correspond to any move
            if first:
                first = False
            else:
                states_actions_returns.append((s, a, G))
            G = r + self.gamma * G
        states_actions_returns.reverse()  # we want it to be in order of state visited
#                 print(states_actions_returns)
        # calculate Q(s,a)
        seen_state_action_pairs = set()
        for s, a, G in states_actions_returns:
            # check if we have already seen s
            # called "first-visit" MC policy evaluation
            sa = (s, a)
            if sa not in seen_state_action_pairs:
                if sa not in self.returns:
                    self.returns[sa] = []
                self.returns[sa].append(G)
                if s not in self.Q:
                    self.Q[s] = np.zeros(env.action_space.n)
                self.Q[s][a] = np.mean(self.returns[sa])
                seen_state_action_pairs.add(sa)
    
        for s in self.Q:
            self.policy[s] = np.argmax(self.Q[s])
        if self.verbose:
#             env.show_policy(self.policy)
            print(self.Q)
            
        return steps
    
    '''
    Interface method
    '''
    def single_iteration_train(self, env_factory, states, verbosity=0):
        if verbosity >= 1:
            print("Updating Value function. Policy improvement.")
        
        steps = 0
#         returns = {}
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
        if s in self.policy:
            return self.policy[s]
        else:
            # if we didn't seen this state before just return rundom_action
            return np.random.choice(action_space)
