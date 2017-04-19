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

class Agent:
    def __init__(self, eps=1.0, alpha=0.5, verbose=False):
        self.eps = eps  # probability of choosing random action instead of greedy
        self.alpha = alpha  # learning rate
        self.state_history = []
        self.verbose = verbose
        self.epoch = 0
        self.random_actions = 0
        self.greedy_actions = 0
        pass     
    
    def setV(self, V):
        self.V = V
    
    def reset_history(self):
        self.state_history = []
        
    def take_action(self, env):
        # choose an action based on epsilon-greedy strategy
        r = np.random.rand()
        eps = float(self.eps) / (self.epoch + 1)
        if r < eps:
            # take a random action
            next_move = np.random.choice(Action.num_actions)
            self.random_actions += 1
            if self.verbose:
                print("Taking a random action " + Action.to_string(next_move))
                print("epsilog: %r < %f" % (r, eps))
        else:
            # choose the best action based on current values of states
            # loop through all possible moves, get their values
            # keep track of the best value
            self.greedy_actions += 1
            next_move = None
            best_value = -100
            for a in range(Action.num_actions):
                # what is the state if we made this move?
                (state, _, _, _) = env.simulate_step(a)
                # Do not count actions which leave agent in same place to avoid bouncing against the wall. 
                # In more advanced algorithms may put small negative reward on each step
                if self.V[state] > best_value and state != env.state:
#                 if self.V[state] > best_value:
                    best_value = self.V[state]
                    next_move = a
            if self.verbose:
                print ("Taking a greedy action " + Action.to_string(next_move))
        # make the move
        state, _, _, _ = env.step(next_move)
#         self.make_move(env, next_move)
        self.state_history.append(state)
        
        # if verbose, draw the grid
        if self.verbose:
            env.show()
            
    def run(self, env, max_steps=10000):
        self.reset()
        actions = []
        steps = 0
        while not env.is_done():
            best_value = -100
            for a in range(Action.num_actions):
                # what is the state if we made this move?
                (state, _, _, _) = env.simulate_step(a)
                # Do not count actions which leave agent in same place to avoid bouncing against the wall. 
                # In more advanced algorithms may put small negative reward on each step
                if self.V[state] > best_value and state != env.state:
#                 if self.V[state] > best_value:
                    best_value = self.V[state]
                    next_move = a
            # make the move
            state, _, _, _ = env.step(next_move)
            self.state_history.append(env.state)
            actions.append(Action.to_string(next_move))
            steps += 1
            if steps > max_steps:
                break
            
        return actions
    
    def update(self, env):
        # we want to BACKTRACK over the states, so that:
        # V(prev_state) = V(prev_state) + alpha*(V(next_state) - V(prev_state))
        # where V(next_state) = reward if it's the most current state
        #
        # NOTE: we ONLY do this at the end of an episode
        # not so for all the algorithms we will study
        reward = env.reward()
        target = reward
#         print(self.state_history)
        self.V[self.state_history.pop()] = target
        for prev in reversed(self.state_history):
            value = self.V[prev] + self.alpha * (target - self.V[prev])
            self.V[prev] = value
            target = value
            
        self.advance_epoch()
        
    def reset(self):
        self.reset_history()
        self.random_actions = 0
        self.greedy_actions = 0
    
    def advance_epoch(self):
        self.epoch += 1

    def single_episode_train(self, env, verbosity=0):
        start_time = timeit.default_timer()
        # reset agent state history. V table doesn't affected by this operation.
        self.reset()
        # add starting point to the history
        self.state_history.append(env.state)
        # if verbose, draw the grid
        if verbosity >= 3:
            env.show()
        # loops until grid is solved
        steps = 0
        while not env.is_done():
            # current player makes a move
            self.take_action(env)
            steps += 1
            # Increase epsilon as workaround to stacking in infinite actions chain
            if steps > env.grid_size * 2 and self.epoch > 1:
                self.epoch /= 2
        # do the value function update
        self.update(env)
        elapsed = timeit.default_timer() - start_time
        if verbosity >= 2:
            print("Solved in %d steps" % len(self.state_history))
            print("Time to solve grid %.3f[ms]" % (elapsed * 1000))
            print("Random actions %d, greedy actions %d" % (self.random_actions, self.greedy_actions))


class Trainer:
    def __init__(self, env_factory):
        self.env_factory = env_factory
        
    def train(self, agent, mini_batch, verbosity=0):
        steps = 0
        if verbosity >= 1:
            print("Train agent for %d iterations." % len(mini_batch))
            start_time = timeit.default_timer()
            
        states = set()
        for s in mini_batch:
            if verbosity >= 3:
                print("\nEpoch #%d" % s)
            # For each episode create new environment so agent will face different starting positions and object locations.
            env = self.env_factory.create_environment(s)
            if env == None:
                continue
            agent.single_episode_train(env, verbosity)
            steps += len(agent.state_history)
            states.update(agent.state_history)      
            if verbosity >= 2:
                print("Episode reward: %f" % env.reward())
        
        elapsed = timeit.default_timer() - start_time

        if verbosity >= 1:
            print("Training time %.3f[ms]" % (elapsed * 1000))
        if verbosity == 0:
            print(" %d" % steps)
        return steps, list(states)

    def evaluate(self, agent, mini_batch, verbosity=0):
        if verbosity >= 2:
            print("Evaluating agent for %d iterations." % len(mini_batch))
            start_time = timeit.default_timer()
        rewards = np.empty(len(mini_batch))
        rewards[:] = np.NaN
        num_iterations = len(mini_batch)
        for i in range(num_iterations):
            env = self.env_factory.create_environment(mini_batch[i])
            if env != None:
                if verbosity >= 3:
                    print()
                    env.show()
                path = agent.run(env, max_steps=env.grid_size)
                
                if verbosity >= 3 or (verbosity >= 2 and env.reward() == REWARD_PIT):
                    print("Failed environment:")
                    env_factory.create_environment(mini_batch[i]).show()
                    print("Agent path")
                    print(path)
                    print("Reward: %.1f" % env.reward())
                rewards[i] = env.reward()
        
        if verbosity >= 1:
            print("Valid states checked %d from total %d" % (num_iterations - len(rewards[np.isnan(rewards)]), num_iterations))
            success = rewards[rewards == REWARD_GOAL].size
            fail = rewards[rewards == REWARD_PIT].size
            hang = rewards[rewards == REWARD_HANG].size
            print("%d ended at goal, %d at pit, %d hanged." % (success, fail, hang))
        if verbosity >= 2:
            elapsed = timeit.default_timer() - start_time
            print("Evaluation time %.3f[ms]" % (elapsed * 1000))
        return np.nanmean(rewards)

class MultiAgentTrainer:
    def __init__(self, env_factory, factor=1, factor_mult=10):
        self.env_factory = env_factory
        self.factor = factor
        self.factor_mult = factor_mult
    
    @staticmethod
    def train_async(agent, env_factory, it_per_agent, i, out_vtable, out_states, verbosity):
        start = timeit.default_timer();
        mini_batch = range(i * it_per_agent, ((i + 1) * it_per_agent))
        trainer = Trainer(env_factory)
        stps, states = trainer.train(agent, mini_batch, verbosity)
        out_states[:len(states)] = states[:] 
        out_vtable[:] = agent.V[:]
        if verbosity >= 1:
            print("trainer%d end %.3f[ms]" % (i, ((timeit.default_timer() - start) * 1000)))
        return stps
    
    def train(self, agents, num_iterations, verbosity=0, async=False):
        steps = 0
        env0 = self.env_factory.create_environment()
        if verbosity >= 1:
            print("Train agents for %d iterations." % env0.num_states)
        start_time = timeit.default_timer()
        
        if num_iterations % len(agents) != 0:
            raise RuntimeError("Number of iterations must be multiply of number of agents")
        
        it_per_agent = int(num_iterations / len(agents))        
        states_per_agent = []
        if async:
            args_list = []
            out_v = [Array('d', np.empty(a.V.size)) for a in agents]
            out_states = [Array('i', [-1] * a.V.size) for a in agents]
            for i in range(len(agents)):
                args_list.append((agents[i], env_factory, it_per_agent, i, out_v[i], out_states[i], verbosity))
            wg = WorkersGroup(len(agents), target=MultiAgentTrainer.train_async, args_list=args_list)
            res = wg.run()
            if verbosity >= 1:
                print("Parallel training end %.3f[ms]" % ((timeit.default_timer() - start_time) * 1000))  
            for i in range(len(agents)):
                steps += res[i]
                agents[i].setV(np.array(out_v[i]))
                temp = np.array(out_states[i])
                states_per_agent.append(temp[temp >= 0])
        else:                   
            trainer = Trainer(self.env_factory)    
            for i in range(len(agents)):
                mini_batch = range(i * it_per_agent, ((i + 1) * it_per_agent))
                stps, states = trainer.train(agents[i], mini_batch, verbosity)
                states_per_agent.append(np.array(states))
                steps += stps
        
        if verbosity >= 1:
            elapsed = timeit.default_timer() - start_time
            print("Multi agents training time %.3f[ms]" % (elapsed * 1000))  
        
        if verbosity >= 1 and len(agents) > 1:
            print("Evaluate agents before mutual update")
            self.evaluate(agents, num_iterations, verbosity=verbosity, async=async)
        
        # Update agents from each other
        for i in range(len(agents)):
            for j in range(len(agents)):
                # Update agent i from all agents j != i
                if i != j:
                    agents[i].V[states_per_agent[j]] = (self.factor * agents[i].V[states_per_agent[j]] + agents[j].V[states_per_agent[j]]) / (self.factor + 1)
    
        self.factor *= self.factor_mult 
        return steps

    @staticmethod
    def evaluate_async(agent, env_factory, it_per_agent, i, verbosity):
        trainer = Trainer(env_factory)
        mini_batch = range(i * it_per_agent, ((i + 1) * it_per_agent))
        reward = trainer.evaluate(agent, mini_batch, verbosity)
        return reward

    def evaluate(self, agents, num_iterations, verbosity=0, async=False):        
        if verbosity >= 1:
            print("Evaluating in parallel, each agent for %d iterations in total." % num_iterations)
            
        start_time = timeit.default_timer()
        
        it_per_agent = int(num_iterations / len(agents))
        rewards = np.empty(len(agents))
        if async:
            args_list = []
            for i in range(len(agents)):
                args_list.append((agents[i], env_factory, it_per_agent, i, verbosity))
            wg = WorkersGroup(len(agents), target=MultiAgentTrainer.evaluate_async, args_list=args_list)
            res = wg.run()
            rewards[:] = res[:]
        else:
            trainer = Trainer(self.env_factory)
            for i in range(len(agents)):
                mini_batch = range(i * it_per_agent, ((i + 1) * it_per_agent))
                reward = trainer.evaluate(agents[i], mini_batch, verbosity)
                rewards[i] = reward
        
        elapsed = timeit.default_timer() - start_time
        if verbosity >= 1:
            print("Total evaluation time %.3f[ms]" % (elapsed * 1000))
        return rewards


def create_agent(verbosity):
    agent = Agent(verbose=(verbosity >= 3))
    V = np.zeros(env.num_states)
    agent.setV(V)
    return agent

np.random.seed(0)
if __name__ == '__main__':
    # Prepare Agent
    verbosity = 1  # 0 - no verbosity; 1 - show prints between episodes; 2 - show agent log
    env_factory = EnvironmentFactory(EnvironmentFactory.EnvironmentType.RandomPlayer)
    env = env_factory.create_environment()

    if verbosity >= 1:
        env.show()
        print()
    
    start_time = timeit.default_timer()

    converged = False
    rewards = []
    total_steps = 0
    total_iterations = 0
    async = False
    if async:
        num_agents = int(math.pow(2, int(math.log(cpu_count(), 2))))
        agents = deque([create_agent(verbosity) for _ in range(num_agents)])
#     agents = deque([create_agent(), create_agent()])
    else:
        agents = deque([create_agent(verbosity)])
    trainer = MultiAgentTrainer(env_factory)
    while not converged:
        steps = trainer.train(agents, env.num_states, verbosity, async=async)
        total_steps += steps
#         max_reward = -100
        print("Evaluate agents to test convergence")
        agents.rotate()
        res = trainer.evaluate(agents, num_iterations=env.num_states, verbosity=verbosity, async=async)
        rewards.append(res.mean())
        if res.max() == REWARD_GOAL:
            converged = True
        
        if verbosity >= 1:
            print()
        total_iterations += 1
    
    elapsed = timeit.default_timer() - start_time
    
    res = []
    if verbosity >= 1:
        for agent in agents:
            res.append(trainer.evaluate([agent], env.num_states, verbosity=verbosity))
    
    print()
    for i in range(len(res)):
        print("Agent %d final reward is %f" % (i, res[i]))
        
    print("Training finished after %d iterations. Total learning steps are %d." % (total_iterations, total_steps))
    print("Total time spent %.3f[s]" % elapsed)
    print("Final mean overal reward %f" % rewards[-1:][0])
    print("Saving V table to vtable.bin")
#     V.tofile('vtable.bin')
    plt.plot(rewards, 'g')
    plt.show()
    print('Done')
