'''
Created on Apr 20, 2017

@author: Yury
'''

import sys
import timeit
from grid_world import *
import numpy as np

SMALL_ENOUGH = 10e-4  # Threshold for convergence
GAMMA = 0.9


class PolicyIterationAgent:
    pass

np.random.seed(0)
if __name__ == '__main__':
    verbosity = 0
    env_factory = EnvironmentFactory(EnvironmentFactory.EnvironmentType.RandomPlayer)
    env = env_factory.create_environment()
    env.show()
    num_states = env.num_states
    # state -> action
    # we will randomly choose the action and update as we learn
    policy = np.random.choice(env.all_actions(), env.num_states)
        
    # initial policy
    print("Initial random policy")
    env.show_policy(policy)
    
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
                    env.show_values(V)
     
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
                for a in env.all_actions():
                    s_prime, r, _, _ = env.simulate_step(a)
                    v = r + GAMMA * V[s_prime]
                    if v > best_value:
                        best_value = v
                        new_a = a
                 
                policy[s] = new_a
                if verbosity >= 3:
                    env.show_policy(policy)
                if new_a != old_a:
                    is_policy_converged = False
                     
        if is_policy_converged:
            break
 
    print("Check policy on random environment")
    env = env_factory.create_environment(np.random.choice(num_states))
    while env == None:
        env = env_factory.create_environment(np.random.choice(num_states))    
    print("values:")
    env.show_values(V)
    print("")
     
    print("optimal  policy:")
    env.show_policy(policy)
    print("iterations to convergence %d" % itt_to_convergence)
                    
