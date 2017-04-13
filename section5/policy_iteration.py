'''
Created on Mar 1, 2017

@author: Yury
'''

import numpy as np
import matplotlib.pyplot as plt
from section5.grid_world import *
from section5.itterative_policy_evaluation import *

SMALL_ENOUGH = 10e-4  # Threshold for convergence
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

if __name__ == '__main__':
    grid = standard_grid()
    
    print('rewards:')
    print_values(grid.rewards, grid)
    
    # state -> action
    # we will randomly choose the action and update as we learn
    policy = {}
    for s in grid.actions.keys():
        policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
        
    # initial policy
    print("initial policy")
    print_policy(policy, grid)
    
    # initialize V(s)
    V = {}
    states=grid.all_states()
    for s in states:
        if s in grid.actions:
            V[s] = np.random.random()
        else:
            V[s] = 0
            
    itt_to_convergence=0
    # repeat until convergence - will break out when policy doesn't change'
    while True:
        itt_to_convergence+=1
        # policy evaluation step
        biggest_change = 0
        for s in states:
            old_v = V[s]
            
            # V(s) has only value if it's not a terminal state 
            if s in policy:
                a = policy[s]
                grid.set_state(s)
                r = grid.move(a)
                V[s] = r + GAMMA * V[grid.current_state()]
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
    
        if biggest_change < SMALL_ENOUGH:
            break
        
        # policy improvement step
        is_policy_converged = True
        for s in states:
            if s in policy:
                old_a = policy[s]
                new_a = None
                best_value = float('-inf')
                
                # loop through all possible actions to find the best current action
                for a in ALL_POSSIBLE_ACTIONS:
                    grid.set_state(s)
                    r = grid.move(a)
                    v = r + GAMMA * V[grid.current_state()]
                    if v > best_value:
                        best_value = v
                        new_a = a
                
                policy[s] = new_a
                if new_a != old_a:
                    is_policy_converged = False
                    
        if is_policy_converged:
            break

    print("")        
    print("values:")
    print_values(V, grid)
    print("")
    
    print("optimal  policy:")
    print_policy(policy, grid)
    print("iterations to convergence %d" % itt_to_convergence)
                    
                    