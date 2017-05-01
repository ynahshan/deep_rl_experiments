'''
Created on Apr 30, 2017

@author: ny
'''

import numpy as np
from rl.environments.grid_world import RandomPlayerEnvironment, RandomGoalAndPlayerEnvironment, Action

class ActionSpace(object):
    def __init__(self, n):
        self.n = n
        
    def sample(self):
        return np.random.choice(self.n)

class BasicGridWorld_v0(object):
    name = 'BasicGridWorld-v0'
    def __init__(self):
        '''
        Constructor
        '''
        self._env = None
        self.action_space = ActionSpace(4)
    
    def step(self, action):
        # Do not check that _env != None to improve performance
        observation, reward, done, info = self._env.step(action)
        if done:
            self._env = None
            
        return observation, reward, done, info 
    
    def reset(self):
        self._env = RandomPlayerEnvironment()
        return self._env.state
    
    def render(self):
        self._env.show()
    
    def close(self):
        self._env = None
    
    def seed(self, seed=None):
        if seed != None:
            np.random.seed(seed)
        return [seed]
    
class BasicGridWorld_v1(object):
    name = 'BasicGridWorld-v1'
    def __init__(self):
        '''
        Constructor
        '''
        self._env = None
        self.action_space = ActionSpace(4)
    
    def step(self, action):
        # Do not check that _env != None to improve performance
        _, reward, done, info = self._env.step(action)
        observation = (self._env.player, self._env.goal)
        if done:
            self._env = None
        
        return observation, reward, done, info 
    
    def reset(self):
        self._env = RandomGoalAndPlayerEnvironment()
        return (self._env.player, self._env.goal)
    
    def render(self):
        self._env.show()
    
    def close(self):
        self._env = None
    
    def seed(self, seed=None):
        if seed != None:
            np.random.seed(seed)
        return [seed]

    def state(self):
        return (self._env.player, self._env.goal)

    def show_policy(self, policy):
        for i in range(self._env.size):
            print("----------------")
            for j in range(self._env.size):
                abs_pos = self._env.cartesian_to_abs((i, j))
                if abs_pos == self._env.wall:
                    symbol = '#'
                elif abs_pos == self._env.goal:
                    symbol = '+'
                elif abs_pos == self._env.pit:
                    symbol = '-'
                else:
                    state = (abs_pos, self._env.goal)
                    if state in policy:
                        action = policy[state]
                        symbol = Action.to_string(action, first_latter=True)
                    else:
                        symbol = '?'

                print((" %s |" % symbol), end='')
            print("")
        print("")

    def show_values(self, V):
        for i in range(self._env.size):
            print("--------------------------------")
            for j in range(self._env.size):
                abs_pos = self._env.cartesian_to_abs((i, j))
                if abs_pos == self._env.wall:
                    symbol = '  #  '
                elif abs_pos == self._env.goal:
                    symbol = '  +  '
                elif abs_pos == self._env.pit:
                    symbol = '  -  '
                else:
                    state = (abs_pos, self._env.goal)
                    if state in V:
                        symbol = "%.2f" % (V[state])
                    else:
                        symbol = '?'
                print((" %s |" % symbol), end='')
            print("")
        print("")