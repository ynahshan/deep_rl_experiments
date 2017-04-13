'''
Created on Mar 21, 2017

@author: Yury
'''

import numpy as np
import matplotlib.pyplot as plt
from section5.grid_world import *
from section5.itterative_policy_evaluation import *

SMALL_ENOUGH = 10e-4  # Threshold for convergence
GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# NOTE: this is only policy evaluation, not optimization

def random_action(a, eps=0.1):
    # we'll  use epsilon-soft to ensure all states are visited
    # what happens if you don'k do this?
    p = np.random.random()
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)

def play_game(grid, policy):
    # return a list of states and  corresponding rewards
    # start at the designated start state
    
    s = (2,0) 
    grid.set_state(s)
    states_and_rewards = [(s,0)] # list of tuples (state, reward)
    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s,r))

    return states_and_rewards

if __name__ == '__main__':
    grid=standard_grid()
    
    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)
    
    policy = {
          (2, 0) : 'U',
          (1, 0) : 'U',
          (0, 0) : 'R',
          (0, 1) : 'R',
          (0, 2) : 'R',
          (1, 2) : 'R',
          (2, 1) : 'R',
          (2, 2) : 'R',
          (2, 3) : 'U',
          }
#     policy = {
#           (0, 0) : 'R',
#           (0, 1) : 'R',
#           (0, 2) : 'R',
#           (1, 0) : 'U',
#           (1, 2) : 'U',
#           (2, 0) : 'U',
#           (2, 1) : 'L',
#           (2, 2) : 'L',
#           (2, 3) : 'L',
#           }
    V={}
    states=grid.all_states()
    for s in states:
        V[s]=0
    
    counter = 0
    for it in range(1000):
        # generate an episode using pi
        counter+=1
        states_and_rewards = play_game(grid, policy)
        # the first (s, r) tuple is the start in 0
        # (since we don'k get a reward) for simply starting the game
        # the last (s, r) tuple is the terminal state and the final reward
        for k in range(len(states_and_rewards) - 1):
            s, _ = states_and_rewards[k]
            s2, r = states_and_rewards[k+1]
            V[s] = V[s] + ALPHA*(r + GAMMA*V[s2] - V[s])
    print("counter = %d" % counter)        
    print("values:")
    print_values(V, grid)
    
    print("policy:")
    print_policy(policy, grid)    
    pass