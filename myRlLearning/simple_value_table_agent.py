'''
Created on Apr 21, 2017

@author: Yury
'''

import sys
import timeit
import numpy as np

class SimpleValueTableAgent:
    def __init__(self, vtable_size, eps=1.0, alpha=0.5, verbose=False):
        self.eps = eps  # probability of choosing random action instead of greedy
        self.alpha = alpha  # learning rate
        self.state_history = []
        self.verbose = verbose
        self.epoch = 0
        self.random_actions = 0
        self.greedy_actions = 0
        self.V = np.zeros(vtable_size)
    
    def reset_history(self):
        self.state_history = []
        
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
        state, _, _, _ = env.step(next_move)
#         self.make_move(env, next_move)
        self.state_history.append(state)
        
        # if verbose, draw the grid
        if self.verbose:
            env.show()
         
    def run(self, env, max_steps=10000):
        self.reset()
        actions = []
        steps = 0
        done = False
        while not done:
            next_move = self.optimal_action(env)
            # make the move
            _, _, done, _ = env.step(next_move)
            self.state_history.append(env.state)
            actions.append(env.action_to_str(next_move))
            steps += 1
            if steps > max_steps:
                break
            
        return actions
    
    def update(self, env):
        # we want to BACKTRACK over the states, so that:
        # V(prev_state) = V(prev_state) + alpha*(V(next_state) - V(prev_state))
        # where V(next_state) = reward if it's the most current state
        #
        # NOTE: we ONLY do this at the end of an episode
        # not so for all the algorithms we will study
        reward = env.reward()
        target = reward
#         print(self.state_history)
        self.V[self.state_history.pop()] = target
        for prev in reversed(self.state_history):
            value = self.V[prev] + self.alpha * (target - self.V[prev])
            self.V[prev] = value
            target = value
            
        self.advance_epoch()
        
    def reset(self):
        self.reset_history()
        self.random_actions = 0
        self.greedy_actions = 0
    
    def advance_epoch(self):
        self.epoch += 1

    def single_episode_train(self, env, verbosity=0):
        start_time = timeit.default_timer()
        # reset agent state history. V table doesn't affected by this operation.
        self.reset()
        # add starting point to the history
        self.state_history.append(env.state)
        # if verbose, draw the grid
        if verbosity >= 3:
            env.show()
        # loops until grid is solved
        steps = 0
        while not env.is_done():
            # current player makes a move
            self.take_action(env)
            steps += 1
            # Increase epsilon as workaround to stacking in infinite actions chain
            if steps > env.grid_size * 2 and self.epoch > 1:
                self.epoch /= 2
        # do the value function update
        self.update(env)
        elapsed = timeit.default_timer() - start_time
        if verbosity >= 2:
            print("Solved in %d steps" % len(self.state_history))
            print("Time to solve grid %.3f[ms]" % (elapsed * 1000))
            print("Random actions %d, greedy actions %d" % (self.random_actions, self.greedy_actions))
    
    '''
    Interface method
    '''    
    def save_model(self, file_name):
        self.V.tofile(file_name)

    '''
    Interface method
    '''
    def single_iteration_train(self, env_factory, states, verbosity=0):
        steps = 0
        for s in states:
            if s % 1000 == 0 and verbosity <= 1:
                sys.stdout.write('.')
                sys.stdout.flush()
            if verbosity >= 3:
                print("\nEpoch #%d" % s)
            # For each episode create new environment so agent will face different starting positions and object locations.
            env = env_factory.create_environment(s)
            if env == None:
                continue
            
            self.single_episode_train(env, verbosity)
            steps += len(self.state_history)
      
            if verbosity >= 2:
                print("Episode reward: %f" % env.reward())
        print()
        return steps

    '''
    Interface method
    '''
    def optimal_action(self, env):
        best_value = float('-inf')
        for a in env.all_actions():
            # what is the state if we made this move?
            (state, _, _, _) = env.simulate_step(a)
            # Do not count actions which leave agent in same place to avoid bouncing against the wall. 
            # In more advanced algorithms may put small negative reward on each step
            if self.V[state] > best_value and state != env.state:
                best_value = self.V[state]
                next_move = a
                
        return next_move