'''
Created on Apr 22, 2017

@author: Yury
'''

import timeit
import matplotlib.pyplot as plt
import numpy as np

from rl.environments.grid_world import GridWorldSolver, EnvironmentFactory, REWARD_GOAL
from rl.agents.monte_carlo_agent import MonteCarloTabularAgent
from rl.agents.policy_iteration_agent import PolicyIterationAgent
from rl.agents.sarsa_agent import SarsaTabularAgent
from rl.agents.qlearning_agent import QLearningTabularAgent

np.random.seed(0)
GAMMA = 0.7
ALPHA = 0.8

def create_agent(env, agent_type, gamma, alpha, verbosity=0):
    class EnvDescriptor:
        def __init__(self, env):
            self.env = env
            self.episod_limit = env.grid_size
        def action_to_str(self, action):
            return env.action_to_str(action)

    if agent_type == "policy_it":
        agent = PolicyIterationAgent(env.num_states, env.all_actions())
    elif agent_type == "monte_carlo":
        agent = MonteCarloTabularAgent(gamma=gamma, env_descriptor=EnvDescriptor(env), verbose=verbosity >= 2)
    elif agent_type == "sarsa":
        agent = SarsaTabularAgent(gamma=gamma, alpha=alpha, env_descriptor=EnvDescriptor(env), verbose=verbosity >= 2)
    elif agent_type == "qlearning":
        agent = QLearningTabularAgent(gamma=gamma, alpha=alpha, env_descriptor=EnvDescriptor(env), verbose=verbosity >= 2)
        
    return agent    

def train_agent(agent_name, env_type, gamma, alpha, verbosity=1):
    env_factory = EnvironmentFactory(env_type)
    env = env_factory.create_environment()
    agent = create_agent(env, agent_name, gamma, alpha, verbosity=verbosity)
    solver = GridWorldSolver(env_factory, agent)
    print("Evaluate %s performance on %s grid world\n" % (agent.__class__.__name__, env.__class__.__name__))
    if verbosity >= 3:
        print("World example:")
        env.show()
        print()
    
    start_time = timeit.default_timer()

    save_model = False
    converged = False
    rewards = []
    total_steps = 0
    total_iterations = 0
    convergence_count = 0
    CONVERGENCE_LIMIT = 10e-4
    CONVERGENCE_STOP_COUNT = 2

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
                convergence_count += 1 
        
        if convergence_count >= CONVERGENCE_STOP_COUNT:
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
        
    return rewards, total_iterations, total_steps

if __name__ == '__main__':
    # Prepare Agent
    verbosity = 0  # 0 - no verbosity; 1 - show prints between episodes; 2 - show agent log
    env_type = EnvironmentFactory.EnvironmentType.RandomPlayer
#     agents = ["policy_it", "monte_carlo", "sarsa", "qlearning"]
    agents = ["monte_carlo", "sarsa", "qlearning"]
#     agents = ["sarsa", "qlearning"]
#     agents = ["monte_carlo"]

    res = {}
    max_it = -1
    for agent in agents:
        res[agent] = train_agent(agent, env_type, gamma=GAMMA, alpha=ALPHA, verbosity=verbosity)
        if res[agent][1] > max_it:
            max_it = res[agent][1]
    
    print()    
    fig, ax = plt.subplots()
    for agent in agents:
        rewards = res[agent][0]
        it = res[agent][1]
        steps = res[agent][2]
        print("Agent: %s, training iterations: %d, total steps %d, final reward %f" % (agent, it, steps, rewards[-1:][0]))
        rewards = np.append(rewards, [rewards[-1:][0]] * (max_it - len(rewards)))
        ax.plot(rewards, label="%s, %d iter, %d steps" % (agent, it, steps))
    
    ax.legend(loc='lower right', shadow=True)
    plt.show()
    print('Done')
