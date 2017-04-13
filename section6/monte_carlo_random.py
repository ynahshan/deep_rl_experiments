'''
Created on Mar 25, 2017

@author: Yury
'''

import numpy as np
import matplotlib.pyplot as plt
from section5.grid_world import *
from section5.itterative_policy_evaluation import *

SMALL_ENOUGH = 10e-4  # Threshold for convergence
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def random_action(a):
    # choose given a with probability 0.5
    # choose some other a' != a with probability 0.5/3
    p = np.random.random()
    if p < 0.5:
        return a
    else:
        tmp = list(ALL_POSSIBLE_ACTIONS)
        tmp.remove(a)
        return np.random.choice(tmp)

def play_game(grid, policy):
    # return a list of states and  corresponding returns
    # reset game to start at random position
    # we need to do this, because given our current deterministic policy
    # we would never end up in a certain states, but we still want to measure the
    start_states = list(grid.actions.keys())
    start_idx=np.random.choice(len(start_states))
    grid.set_state(start_states[start_idx])
    
    s = grid.current_state()
    states_and_rewards = [(s,0)] # list of tupples (state, reward)
    while not grid.game_over():
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        states_and_rewards.append((s,r))
    # calculate the returns by working backward from the terminal state
    G=0
    states_and_returns=[]
    first=True
    for s, r in reversed(states_and_rewards):
        # the value of the terminal state is 0 by definition
        # we should ignore the first state we encounter
        # and ignore the last  G, which is meaningless since it doesn'k corresponds
        if first:
            first = False
        else:
            states_and_returns.append((s, G))
        G=r+GAMMA*G
    states_and_returns.reverse() # we want it to be in order of states visited
    return states_and_returns

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
#           (1, 2) : 'R',
#           (2, 1) : 'R',
#           (2, 2) : 'R',
#           (2, 3) : 'U',
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
    V={}
    returns = {}
    states=grid.all_states()
    for s in states:
        if s in grid.actions:
            returns[s] = []
        else:
            # terminal state or state we can'k otherwise get to
            V[s]=0
            
    # repeat until convergence
    for k in range(1000):
        #generate an episode using pi
        states_and_returns = play_game(grid, policy)
        seen_states=set()
        for s, G in states_and_returns:
            # check if we have already seen s
            # called "first-visit" MC policy evaluation
            if s not in seen_states:
                returns[s].append(G)
                V[s]=np.mean(returns[s])
                seen_states.add(s)
    
    print("values:")
    print_values(V, grid)
    
    print("policy:")
    print_policy(policy, grid)    
    pass
