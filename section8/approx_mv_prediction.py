'''
Created on Mar 25, 2017

@author: Yury
'''
import numpy as np
import matplotlib.pyplot as plt
from section5.grid_world import *
from section5.itterative_policy_evaluation import *

from section6.monte_carlo_random import random_action, play_game, SMALL_ENOUGH, GAMMA

np.random.seed(0)

LEARNING_RATE = 0.001

if __name__ == '__main__':
    grid=standard_grid()
    
    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)
    
#     policy = {
#           (2, 0) : 'U',
#           (1, 0) : 'U',
#           (0, 0) : 'R',
#           (0, 1) : 'R',
#           (0, 2) : 'R',
#           (1, 2) : 'U',
#           (2, 1) : 'L',
#           (2, 2) : 'U',
#           (2, 3) : 'L',
#           }
    policy = {
          (0, 0) : 'R',
          (0, 1) : 'R',
          (0, 2) : 'R',
          (1, 0) : 'U',
          (1, 2) : 'U',
          (2, 0) : 'U',
          (2, 1) : 'L',
          (2, 2) : 'U',
          (2, 3) : 'L',
          }
    
    # initialize Theta
    # # our model is V_hat = theta.dot(x)
    # where x = [row, col, row*col, 1] - 1 for bias term
    theta = np.random.randn(4) / 2
    def s2x(s):
        return np.array([s[0] - 1, s[1] - 1.5, s[0]*s[1]-0, 1])
    
    # repeat until convergence
    deltas = []
    k = 1.0
    
    for it in range(20000):
        if it % 10 == 0:
            k+=0.01
        alpha = LEARNING_RATE/k
        # generate an episode using pi
        biggest_change = 0
        states_and_returns = play_game(grid, policy)
        seen_states = set()
        for s, G in states_and_returns:
            # check if we have already seen s
            # called "first-visit" MC policy evaluation
            if s not in seen_states:
                old_theta = theta.copy()
                x = s2x(s)
                V_hat = theta.dot(x)
                # grad(V_hat) wrt theta = x
                theta += alpha*(G - V_hat)*x
                biggest_change = max(biggest_change, np.abs(old_theta - theta).sum())
                seen_states.add(s)
        deltas.append(biggest_change)
        
    plt.plot(deltas)
    plt.show()
    
    V = {}
    states = grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = theta.dot(s2x(s))
        else:
            # terminal state or state we can'k otherwise get to
            V[s] = 0
        
    print("values:")
    print_values(V, grid)
    
    print("policy:")
    print_policy(policy, grid) 
    pass