'''
Created on Apr 22, 2017

@author: Yury
'''

import timeit
import matplotlib.pyplot as plt
import numpy as np

from grid_world import GridWorldSolver, EnvironmentFactory, REWARD_GOAL
from simple_value_table_agent import SimpleValueTableAgent
from policy_iteration_agent import PolicyIterationAgent
from monte_carlo_agent import MonteCarloAgent
from sarsa_agent import SarsaAgent

agents = ["simple", "policy_it", "monte_carlo", "sarsa"]

CONVERGENCE_LIMIT = 10e-4

def create_agent(env, agent_type, verbosity=0):
    if agent_type == "simple":
        agent = SimpleValueTableAgent(env.num_states)
    elif agent_type == "policy_it":
        agent = PolicyIterationAgent(env.num_states, env.all_actions())
    elif agent_type == "monte_carlo":
        agent = MonteCarloAgent(eps=1.0, gamma=0.8, verbose=verbosity >= 2)
    elif agent_type == "sarsa":
        agent = SarsaAgent(verbose=verbosity >= 2)
        
    return agent    

np.random.seed(0)
if __name__ == '__main__':
    # Prepare Agent
    verbosity = 1  # 0 - no verbosity; 1 - show prints between episodes; 2 - show agent log
    env_factory = EnvironmentFactory(EnvironmentFactory.EnvironmentType.RandomPlayer)
    env = env_factory.create_environment()
    agent = create_agent(env, agents[3], verbosity=verbosity)
    solver = GridWorldSolver(env_factory, agent)
    print("Evaluate %s performance on %s grid world\n" % (agent.__class__.__name__, env.__class__.__name__))
    if verbosity >= 1:
        print("World example:")
        env.show()
        print()
    
    start_time = timeit.default_timer()

    save_model = True
    converged = False
    rewards = []
    total_steps = 0
    total_iterations = 0
    while not converged:
        print("[%d] Train agent with all possible states" % total_iterations)
        steps = solver.train(range(env.num_states), verbosity)
        total_steps += steps
        print("[%d] Evaluate agent to test convergence" % total_iterations)
        res = solver.evaluate(range(env.num_states), verbosity)
        print("Reward: %f" % res)
        rewards.append(res.mean())
        if res.max() == REWARD_GOAL:
            converged = True
        if total_iterations > 0:
            diff = rewards[total_iterations] - rewards[total_iterations - 1]
            if np.abs(diff) < CONVERGENCE_LIMIT:
                converged = True 
        
        if verbosity >= 0:
            print()
        total_iterations += 1
#         break
    
    elapsed = timeit.default_timer() - start_time
    
    print("Evaluation finished.")
    print("Agent final reward is %f" % res)        
    print("Training finished after %d iterations. Total learning steps are %d." % (total_iterations, total_steps))
    print("Total time spent %.3f[s]" % elapsed)
    print("Final mean overal reward %f" % rewards[-1:][0])
    print("Saving V table to vtable.bin")
    if save_model:
        agent.save_model('vtable.bin')

    plt.plot(rewards, 'g')
    plt.show()
    print('Done')
