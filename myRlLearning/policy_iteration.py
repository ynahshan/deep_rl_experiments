'''
Created on Apr 20, 2017

@author: Yury
'''

import sys
import timeit

sys.path.append('../')
import matplotlib.pyplot as plt
from grid_world import *
import numpy as np
from utils.threading.worker import WorkersGroup
from collections import deque
from multiprocessing import Array, cpu_count
import math

SMALL_ENOUGH = 10e-4  # Threshold for convergence
GAMMA = 0.9

def print_policy(policy, env):
    for i in range(env.size):
        print("----------------")
        for j in range(env.size):
            abs_pos = env.cartesian_to_abs((i, j))
            if abs_pos == env.wall:
                symbol = '#'
            elif abs_pos == env.goal:
                symbol = '+'
            elif abs_pos == env.pit:
                symbol = '-'
            else:  
                state = env.player_abs_to_state(abs_pos)
                action = policy[state]
                symbol = Action.to_string(action, first_latter=True)
            print((" %s |" % symbol), end='')
        print("")
    print("")

def print_values(V, env):        
    for i in range(env.size):
        print("--------------------------------")
        for j in range(env.size):
            abs_pos = env.cartesian_to_abs((i, j))
            if abs_pos == env.wall:
                symbol = '  #  '
            elif abs_pos == env.goal:
                symbol = '  +  '
            elif abs_pos == env.pit:
                symbol = '  -  '
            else:  
                state = env.player_abs_to_state(abs_pos)
                symbol = "%.2f" % (V[state])
            print((" %s |" % symbol), end='')
        print("")
    print("")

ALL_ACTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

np.random.seed(0)
if __name__ == '__main__':
    verbosity = 0
    env_factory = EnvironmentFactory(EnvironmentFactory.EnvironmentType.AllRandom)
    env = env_factory.create_environment()
    env.show()
    num_states = env.num_states
    # state -> action
    # we will randomly choose the action and update as we learn
    policy = np.random.choice(ALL_ACTIONS, env.num_states)
        
    # initial policy
    print("Initial random policy")
    print_policy(policy, env)
    
    # initialize V(s)
    V = np.zeros(num_states)
             
    itt_to_convergence = 0
    # repeat until convergence - will break out when policy doesn't change'
    while True:
        itt_to_convergence += 1
        # policy evaluation step
        biggest_change = 0
        if verbosity >= 3:
            print(itt_to_convergence)
        for s in range(num_states):
            old_v = V[s]
            env = env_factory.create_environment(s)
            # V(s) has only value if it's not a terminal state 
            if env != None:
                a = policy[s]
                s_prime, r, _, _ = env.simulate_step(a)
                V[s] = r + GAMMA * V[s_prime]
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
                if verbosity >= 3:
                    print_values(V, env)
     
        if biggest_change < SMALL_ENOUGH:
            break
        
        # policy improvement step
        is_policy_converged = True
        for s in range(num_states):
            env = env_factory.create_environment(s)
            if env != None:
                old_a = policy[s]
                new_a = None
                best_value = float('-inf')
                # loop through all possible actions to find the best current action
                for a in ALL_ACTIONS:
                    s_prime, r, _, _ = env.simulate_step(a)
                    v = r + GAMMA * V[s_prime]
                    if v > best_value:
                        best_value = v
                        new_a = a
                 
                policy[s] = new_a
                if verbosity >= 3:
                    print_policy(policy, env)
                if new_a != old_a:
                    is_policy_converged = False
                     
        if is_policy_converged:
            break
 
    print("Check policy on random environment")
    env = env_factory.create_environment(np.random.choice(num_states))
    while env == None:
        env = env_factory.create_environment(np.random.choice(num_states))    
    print("values:")
    print_values(V, env)
    print("")
     
    print("optimal  policy:")
    print_policy(policy, env)
    print("iterations to convergence %d" % itt_to_convergence)
                    
