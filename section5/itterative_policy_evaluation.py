'''
Created on Feb 26, 2017

@author: Yury
'''

import numpy as np
import matplotlib.pyplot as plt
from section5.grid_world import *

SMALL_ENOUGH = 10e-3  # Threshold for convergence

def print_values(V, g):
    for i in range(g.height):
        print("----------------------------------")
        for j in range(g.width):
            v = V.get((i, j), 0)
            if v >= 0:
                print((" %.2f|" % v), end='')
            else:
                print (("%.2f|" % v), end='')
        print("")

def print_policy(P, g):
    for i in range(g.height):
        print("----------------------------------")
        for j in range(g.width):
            a = P.get((i, j), ' ')
            print((" %s |" % a), end='')
        print("")
        
if __name__ == '__main__':
    # iterative policy evaluation, given a policy find a value function V(s)
    # we are modeling here p(a|s) = uniform
    grid = standard_grid()
    
    # states will be positions (i, j)
    states = grid.all_states()
    
    ### Uniform random actions ###
    # initialize V(s) = 0
    V = {}
    for s in states:
        V[s] = 0
        
    gamma = 0.9
    # repeat until convergence
    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]
            
            # V(s) has only value if it's not a terminal state
            if s in grid.actions:
                new_v = 0  # accumulator for v
                p_a = 1.0 / len(grid.actions[s])
                for a in grid.actions[s]:
                    grid.set_state(s)
                    r = grid.move(a)
                    new_v += p_a * (r + gamma * V[grid.current_state()])
                V[s] = new_v
                biggest_change = max(biggest_change, np.abs(old_v - new_v))
            
        if biggest_change < SMALL_ENOUGH:
            break
    
    print("values for uniform random actions:")   
    print_values(V, grid)
    print("\n\n")
    
    # fixed policy
    policy = {
              (0, 0) : 'R',
              (0, 1) : 'R',
              (0, 2) : 'R',
              (1, 0) : 'U',
              (1, 2) : 'U',
              (2, 0) : 'U',
              (2, 1) : 'L',
              (2, 2) : 'L',
              (2, 3) : 'L',
              }
    print("policy that we evaluate")
    print_policy(policy, grid)
    print("")
    
    V = {}
    for s in states:
        V[s] = 0
        
    gamma = 0.9
    # repeat until convergence
    while True:
        biggest_change = 0
        for s in states:
            old_v = V[s]
            
            # V(s) has only value if it's not a terminal state 
            if s in policy:
                a = policy[s]
                grid.set_state(s)
                r = grid.move(a)
                V[s] = r + gamma * V[grid.current_state()]
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
    
        if biggest_change < SMALL_ENOUGH:
            break
    
    print("values for fixed policy:")
    print_values(V, grid)

    pass
