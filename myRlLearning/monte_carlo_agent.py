'''
Created on Apr 23, 2017

@author: Yury
'''

import sys
import timeit
import numpy as np

class MonteCarloAgent(object):
    def __init__(self, eps=1.0, gamma=0.9, verbose=False):
        self.eps = eps
        self.gamma = gamma
        self.epoch = 0
        self.policy = {}
        self.Q = {}
        self.random_actions = 0
        self.greedy_actions = 0
        self.verbose = verbose

    def take_action(self, env):
        # choose an action based on epsilon-greedy strategy
        r = np.random.rand()
        eps = float(self.eps) / (self.epoch + 1)
        if r < eps:
            # take a random action
            next_move = np.random.choice(len(env.all_actions()))
            self.random_actions += 1
            if self.verbose:
                print("Taking a random action " + env.action_to_str(next_move))
                print("epsilog: %r < %f" % (r, eps))
        else:
            # choose the best action based on current values of states
            # loop through all possible moves, get their values
            # keep track of the best value
            self.greedy_actions += 1
            next_move = self.optimal_action(env)
            if self.verbose:
                print ("Taking a greedy action " + env.action_to_str(next_move))
        # make the move
        state, r, done, _ = env.step(next_move)
        
        # if verbose, draw the grid
        if self.verbose:
            env.show()
            
        return state, r, done, next_move

    def single_episode_exploration(self, env):
#         start_time = timeit.default_timer()
        # loops until grid is solved
#         steps = 0
        states_actions_rewards = []
        s, r, done, a = self.take_action(env)
        states_actions_rewards.append((s, a, 0))
        done = False
        while not done:
            if done:
                states_actions_rewards.append((s, None, r))
            else:
                s, r, done, a = self.take_action(env)
                states_actions_rewards.append((s, a, r))
        if self.verbose:
            print("Episode finished\n")
        self.epoch += 1
#             steps += 1
#             # Increase epsilon as workaround to stacking in infinite actions chain
#             if steps > env.grid_size * 2 and self.epoch > 1:
#                 self.epoch /= 2
            
#         elapsed = timeit.default_timer() - start_time
#         if verbosity >= 2:
#             print("Solved in %d steps" % len(self.state_history))
#             print("Time to solve grid %.3f[ms]" % (elapsed * 1000))
#             print("Random actions %d, greedy actions %d" % (self.random_actions, self.greedy_actions))

        return states_actions_rewards
            
    def display_functions(self, env):
#         env.show_values(self.V)
        env.show_policy(self.policy, full=False)
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
        
        returns = {}
        for s in states:
            if s % 1000 == 0 and verbosity <= 1:
                sys.stdout.write('.')
                sys.stdout.flush()

            # Doing create_environment(s) instead similar to exploring starts method
            env = env_factory.create_environment()
            # V(s) has only value if it's not a terminal state 
            if env != None:
                states_actions_rewards = self.single_episode_exploration(env)
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
                
                # calculate Q(s,a)
                seen_state_action_pairs = set()
                for s, a, G in states_actions_returns:
                    # check if we have already seen s
                    # called "first-visit" MC policy evaluation
                    sa = (s, a)
                    if sa not in seen_state_action_pairs:
                        if sa not in returns:
                            returns[sa] = []
                        returns[sa].append(G)
                        if s not in self.Q:
                            self.Q[s] = np.zeros(len(env.all_actions()))
                        self.Q[s][a] = np.mean(returns[sa])
                        seen_state_action_pairs.add(sa)
            
                self.policy[s] = np.argmax(self.Q[s])
                if self.verbose:
                    env.show_policy(self.policy, full=False)
                    print(self.Q)
                    
        print()
        return len(states)

    '''
    Interface method
    '''
    def optimal_action(self, env):
        s = env.state
        if s in self.policy:
            return self.policy[s]
        else:
            # if we didn't seen this state before just do random action
            return np.random.choice(env.all_actions())
