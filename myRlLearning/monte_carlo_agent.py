'''
Created on Apr 23, 2017

@author: Yury
'''

import sys
import timeit
import numpy as np

class MonteCarloAgent(object):
    def __init__(self, eps = 1.0, gamma=0.9):
        self.eps = eps
        self.gamma = gamma
        self.epoch = 0
        self.policy = {}

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

    def single_episode_exploration(self, env, verbosity=0):
#         start_time = timeit.default_timer()

        # if verbose, draw the grid
        if verbosity >= 3:
            env.show()
        # loops until grid is solved
#         steps = 0
        states_actions_rewards = []
        done = False
        while not done:
            s_prime, r, done, a = self.take_action(env)
            if done:
                states_actions_rewards.append((s_prime, None, r))
            else:
                states_actions_rewards.append((s_prime, a, r))
                
#             steps += 1
#             # Increase epsilon as workaround to stacking in infinite actions chain
#             if steps > env.grid_size * 2 and self.epoch > 1:
#                 self.epoch /= 2
            
#         elapsed = timeit.default_timer() - start_time
#         if verbosity >= 2:
#             print("Solved in %d steps" % len(self.state_history))
#             print("Time to solve grid %.3f[ms]" % (elapsed * 1000))
#             print("Random actions %d, greedy actions %d" % (self.random_actions, self.greedy_actions))
            
    def display_functions(self, env):
#         env.show_values(self.V)
#         env.show_policy(self.policy)
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
        for s in states:
            if s % 1000 == 0 and verbosity <= 1:
                sys.stdout.write('.')
                sys.stdout.flush()

            # Doing create_environment(s) instead similar to exploring starts method
            env = env_factory.create_environment()
            # V(s) has only value if it's not a terminal state 
            if env != None:
                a = self.policy[s]
                s_prime, r, _, _ = env.simulate_step(a)
                self.V[s] = r + self.gamma * self.V[s_prime]
                if verbosity >= 3:
                    env.show_values(V)
            

        print()
        if verbosity >= 1:
            print("Policy evaluation")
        for s in states:
            if s % 1000 == 0 and verbosity <= 1:
                sys.stdout.write('.')
                sys.stdout.flush()
            env = env_factory.create_environment(s)
            if env != None:
                best_value = float('-inf')
                # loop through all possible actions to find the best current action.
                # It is basically the Q function evaluation for state s.
                for a in env.all_actions():
                    s_prime, r, _, _ = env.simulate_step(a)
                    v = r + self.gamma * self.V[s_prime]
                    if v > best_value:
                        best_value = v
                        best_action = a
                 
                self.policy[s] = best_action
                if verbosity >= 3:
                    env.show_policy(policy)
                    
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