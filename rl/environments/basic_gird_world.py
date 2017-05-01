'''
Created on Apr 30, 2017

@author: ny
'''

import numpy as np
from rl.environments.grid_world import RandomPlayerEnvironment, RandomGoalAndPlayerEnvironment

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
        self.__env = None
        self.action_space = ActionSpace(4)
    
    def step(self, action):
        # Do not check that __env != None to improve performance
        observation, reward, done, info = self.__env.step(action)
        if done:
            self.__env = None
            
        return observation, reward, done, info 
    
    def reset(self):
        self.__env = RandomPlayerEnvironment()
        return self.__env.state
    
    def render(self):
        self.__env.show()
    
    def close(self):
        self.__env = None
    
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
        self.__env = None
        self.action_space = ActionSpace(4)
    
    def step(self, action):
        # Do not check that __env != None to improve performance
        _, reward, done, info = self.__env.step(action)
        observation = (self.__env.player, self.__env.goal)
        if done:
            self.__env = None
        
        return observation, reward, done, info 
    
    def reset(self):
        self.__env = RandomGoalAndPlayerEnvironment()
        return (self.__env.player, self.__env.goal)
    
    def render(self):
        self.__env.show()
    
    def close(self):
        self.__env = None
    
    def seed(self, seed=None):
        if seed != None:
            np.random.seed(seed)
        return [seed]