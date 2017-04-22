'''
Created on Apr 15, 2017

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
from simple_value_table_agent import SimpleValueTableAgent

# class MultiAgentTrainer:
#     def __init__(self, env_factory, factor=1, factor_mult=10):
#         self.env_factory = env_factory
#         self.factor = factor
#         self.factor_mult = factor_mult
#     
#     @staticmethod
#     def train_async(agent, env_factory, it_per_agent, i, out_vtable, verbosity):
#         start = timeit.default_timer();
#         mini_batch = range(i * it_per_agent, ((i + 1) * it_per_agent))
#         stps = agent.train(agent, mini_batch, verbosity) 
#         out_vtable[:] = agent.V[:]
#         if verbosity >= 1:
#             print("trainer%d end %.3f[ms]" % (i, ((timeit.default_timer() - start) * 1000)))
#         return stps
#     
#     def train(self, agents, num_iterations, verbosity=0, async=False):
#         steps = 0
#         env0 = self.env_factory.create_environment()
#         if verbosity >= 1:
#             print("Train agents for %d iterations." % env0.num_states)
#         start_time = timeit.default_timer()
#         
#         if num_iterations % len(agents) != 0:
#             raise RuntimeError("Number of iterations must be multiply of number of agents")
#         
#         it_per_agent = int(num_iterations / len(agents))        
#         states_per_agent = []
#         if async:
#             args_list = []
#             out_v = [Array('d', np.empty(a.V.size)) for a in agents]
#             out_states = [Array('i', [-1] * a.V.size) for a in agents]
#             for i in range(len(agents)):
#                 args_list.append((agents[i], env_factory, it_per_agent, i, out_v[i], verbosity))
#             wg = WorkersGroup(len(agents), target=MultiAgentTrainer.train_async, args_list=args_list)
#             res = wg.run()
#             if verbosity >= 1:
#                 print("Parallel training end %.3f[ms]" % ((timeit.default_timer() - start_time) * 1000))  
#             for i in range(len(agents)):
#                 steps += res[i]
#                 agents[i].setV(np.array(out_v[i]))
#                 temp = np.array(out_states[i])
#                 states_per_agent.append(temp[temp >= 0])
#         else: 
#             for i in range(len(agents)):
#                 mini_batch = range(i * it_per_agent, ((i + 1) * it_per_agent))
#                 stps = agents[i].train(mini_batch, verbosity)
# #                 states_per_agent.append(np.array(states))
#                 steps += stps
#         
#         if verbosity >= 1:
#             elapsed = timeit.default_timer() - start_time
#             print("Multi agents training time %.3f[ms]" % (elapsed * 1000))  
#         
#         if verbosity >= 1 and len(agents) > 1:
#             print("Evaluate agents before mutual update")
#             self.evaluate(agents, num_iterations, verbosity=verbosity, async=async)
#         
#         # Update agents from each other
#         for i in range(len(agents)):
#             for j in range(len(agents)):
#                 # Update agent i from all agents j != i
#                 if i != j:
#                     agents[i].V[states_per_agent[j]] = (self.factor * agents[i].V[states_per_agent[j]] + agents[j].V[states_per_agent[j]]) / (self.factor + 1)
#      
#         self.factor *= self.factor_mult 
#         return steps
# 
#     @staticmethod
#     def evaluate_async(agent, env_factory, it_per_agent, i, verbosity):
#         mini_batch = range(i * it_per_agent, ((i + 1) * it_per_agent))
#         reward = agent.evaluate(agent, mini_batch, verbosity)
#         return reward
# 
#     def evaluate(self, agents, num_iterations, verbosity=0, async=False):        
#         if verbosity >= 1:
#             print("Evaluating in parallel, each agent for %d iterations in total." % num_iterations)
#             
#         start_time = timeit.default_timer()
#         
#         it_per_agent = int(num_iterations / len(agents))
#         rewards = np.empty(len(agents))
#         if async:
#             args_list = []
#             for i in range(len(agents)):
#                 args_list.append((agents[i], env_factory, it_per_agent, i, verbosity))
#             wg = WorkersGroup(len(agents), target=MultiAgentTrainer.evaluate_async, args_list=args_list)
#             res = wg.run()
#             rewards[:] = res[:]
#         else:
#             for i in range(len(agents)):
#                 mini_batch = range(i * it_per_agent, ((i + 1) * it_per_agent))
#                 reward = agents[i].evaluate(mini_batch, verbosity)
#                 rewards[i] = reward
#         
#         elapsed = timeit.default_timer() - start_time
#         if verbosity >= 1:
#             print("Total evaluation time %.3f[ms]" % (elapsed * 1000))
#         return rewards


def create_agent(env_factory, verbosity):
    agent = SimpleValueTableAgent(env_factory.create_environment().num_states, verbose=(verbosity >= 3))    
    gda = GridWorldSolver(env_factory, agent)
    return gda

np.random.seed(0)
if __name__ == '__main__':
    # Prepare Agent
    verbosity = 1  # 0 - no verbosity; 1 - show prints between episodes; 2 - show agent log
    env_factory = EnvironmentFactory(EnvironmentFactory.EnvironmentType.RandomPlayerAndGoal)
    env = env_factory.create_environment()

    if verbosity >= 1:
        env.show()
        print()
    
    start_time = timeit.default_timer()

    converged = False
    rewards = []
    total_steps = 0
    total_iterations = 0
    agent = create_agent(env_factory, verbosity)
    while not converged:
        steps = agent.train(range(env.num_states), verbosity)
        total_steps += steps
        print("Evaluate agents to test convergence")
        res = agent.evaluate(range(env.num_states), verbosity)
        rewards.append(res.mean())
        if res.max() == REWARD_GOAL:
            converged = True
        
        if verbosity >= 1:
            print()
        total_iterations += 1
    
    elapsed = timeit.default_timer() - start_time
    
    print()
    print("Agent final reward is %f" % res)        
    print("Training finished after %d iterations. Total learning steps are %d." % (total_iterations, total_steps))
    print("Total time spent %.3f[s]" % elapsed)
    print("Final mean overal reward %f" % rewards[-1:][0])
    print("Saving V table to vtable.bin")
#     V.tofile('vtable.bin')
    plt.plot(rewards, 'g')
    plt.show()
    print('Done')
