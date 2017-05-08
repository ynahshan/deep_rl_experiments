'''
Created on Apr 30, 2017

@author: ny
'''

 
from __future__ import print_function

import numpy as np
from rl.environments.grid_world import RandomPlayerEnvironment, RandomGoalAndPlayerEnvironment, Action

class ActionSpace(object):
    def __init__(self, n):
        self.n = n
        
    def sample(self):
        return np.random.choice(self.n)

class ObservationSpace(object):
    def __init__(self, env):
        self.env = env

    def sample(self):
        return self.env._obs_sample()

class GridWorldBase(object):
    def __init__(self):
        '''
        Constructor
        '''
        self._env = None
        self.action_space = ActionSpace(4)
        self.observation_space = ObservationSpace(self)
    
    def render(self):
        self._env.show()
    
    def close(self):
        self._env = None
    
    def seed(self, seed=None):
        if seed != None:
            np.random.seed(seed)
        return [seed]
      
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
                    state = self.state_pos(abs_pos)
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
                    state = self.state_pos(abs_pos)
                    if state in V:
                        symbol = "%.2f" % (V[state])
                    else:
                        symbol = '?'
                print((" %s |" % symbol), end='')
            print("")
        print("")

class BasicGridWorld_v0(GridWorldBase):
    name = 'BasicGridWorld-v0'
    
    def step(self, action):
        # Do not check that _env != None to improve performance
        observation, reward, done, info = self._env.step(action)
        if done:
            self._env = None
            
        return observation, reward, done, info 
    
    def reset(self):
        self._env = RandomPlayerEnvironment()
        return self._env.state

    def state_pos(self, pos):
        return pos
    
    def state(self):
        return self._env.player

    def _obs_sample(self):
        return np.random.choice(RandomPlayerEnvironment.grid_size)
    
class BasicGridWorld_v1(GridWorldBase):
    name = 'BasicGridWorld-v1'
    
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

    def state_pos(self, pos):
        return (pos, self._env.goal)
    
    def state(self):
        return (self._env.player, self._env.goal)

    def _obs_sample(self):
        return (np.random.choice(RandomGoalAndPlayerEnvironment.grid_size), np.random.choice(RandomGoalAndPlayerEnvironment.grid_size))