'''
Created on Apr 15, 2017

@author: Yury
'''

import sys
import timeit

import matplotlib.pyplot as plt
from myRlLearning.grid_world import EnvironmentFactory, REWARD_GOAL, REWARD_HANG, REWARD_PIT
import numpy as np
from utils.threading.worker import WorkersGroup
from collections import deque

class Action:
    num_actions = 4
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    @staticmethod
    def to_string(a):
        if a == Action.UP:
            return 'UP'
        elif a == Action.DOWN:
            return 'DOWN'
        elif a == Action.LEFT:
            return 'LEFT'
        elif a == Action.RIGHT:
            return 'RIGHT'
        else:
            return 'n/a'

class Agent:
    def __init__(self, eps=1, alpha=0.5, verbose=False):
        self.eps = eps  # probability of choosing random action instead of greedy
        self.alpha = alpha  # learning rate
        self.state_history = []
        self.verbose = verbose
        self.epoch = 0
        self.random_actions = 0
        self.greedy_actions = 0
        pass
    
    def copy(self):
        agent = Agent(self.eps, self.alpha, self.verbose)
        agent.epoch = self.epoch
        agent.random_actions = self.random_actions
        agent.greedy_actions = self.greedy_actions
        agent.state_history = self.state_history[:]
        agent.V = self.V.copy()
        return agent        
    
    def setV(self, V):
        self.V = V
    
    def reset_history(self):
        self.state_history = []
        
    def take_action(self, env):
        # choose an action based on epsilon-greedy strategy
        r = np.random.rand()
        eps = self.eps / (self.epoch + 1)
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
                state = self.imitate_move(env, a)
                # Do not count actions which leave agent in same place to avoid bouncing against the wall. 
                # In more advanced algorithms may put small negative reward on each step
                if self.V[state] > best_value and state != env.state:
#                 if self.V[state] > best_value:
                    best_value = self.V[state]
                    next_move = a
            if self.verbose:
                print ("Taking a greedy action " + Action.to_string(next_move))
        # make the move
        self.make_move(env, next_move)
        self.state_history.append(env.state)
        
        # if verbose, draw the grid
        if self.verbose:
            env.show()
            
    def run(self, env, max_steps=10000):
        self.reset(env)
        actions = []
        steps = 0
        while not env.is_done():
            best_value = -100
            for a in range(Action.num_actions):
                # what is the state if we made this move?
                state = self.imitate_move(env, a)
                # Do not count actions which leave agent in same place to avoid bouncing against the wall. 
                # In more advanced algorithms may put small negative reward on each step
                if self.V[state] > best_value and state != env.state:
#                 if self.V[state] > best_value:
                    best_value = self.V[state]
                    next_move = a
            # make the move
            self.make_move(env, next_move)
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
        
    def reset(self, env):
        self.reset_history()
        self.location = (int(env.player_starting_point / env.size), env.player_starting_point % env.size)
        self.random_actions = 0
        self.greedy_actions = 0
    
    def advance_epoch(self):
        self.epoch += 1
    
    def make_move(self, env, action):    
        # up (row - 1)
        if action == Action.UP:
            new_loc = (self.location[0] - 1, self.location[1])
            if (new_loc != env.wall_cartesian):
                if ((np.array(new_loc) <= (env.size - 1, env.size - 1)).all() and (np.array(new_loc) >= (0, 0)).all()):
                    env.update_state(new_loc)
                    self.location = new_loc
        # down (row + 1)
        elif action == Action.DOWN:
            new_loc = (self.location[0] + 1, self.location[1])
            if (new_loc != env.wall_cartesian):
                if ((np.array(new_loc) <= (3, 3)).all() and (np.array(new_loc) >= (0, 0)).all()):
                    env.update_state(new_loc)
                    self.location = new_loc
        # left (column - 1)
        elif action == Action.LEFT:
            new_loc = (self.location[0], self.location[1] - 1)
            if (new_loc != env.wall_cartesian):
                if ((np.array(new_loc) <= (3, 3)).all() and (np.array(new_loc) >= (0, 0)).all()):
                    env.update_state(new_loc)
                    self.location = new_loc
        # right (column + 1)
        elif action == Action.RIGHT:
            new_loc = (self.location[0], self.location[1] + 1)
            if (new_loc != env.wall_cartesian):
                if ((np.array(new_loc) <= (3, 3)).all() and (np.array(new_loc) >= (0, 0)).all()):
                    env.update_state(new_loc)
                    self.location = new_loc
    
    def imitate_move(self, env, action):
        state = env.state
        # up (row - 1)
        if action == Action.UP:
            new_loc = (self.location[0] - 1, self.location[1])
            if (new_loc != env.wall_cartesian):
                if ((np.array(new_loc) <= (3, 3)).all() and (np.array(new_loc) >= (0, 0)).all()):
                    state = env.get_state(new_loc)
        # down (row + 1)
        elif action == Action.DOWN:
            new_loc = (self.location[0] + 1, self.location[1])
            if (new_loc != env.wall_cartesian):
                if ((np.array(new_loc) <= (3, 3)).all() and (np.array(new_loc) >= (0, 0)).all()):
                    state = env.get_state(new_loc)
        # left (column - 1)
        elif action == Action.LEFT:
            new_loc = (self.location[0], self.location[1] - 1)
            if (new_loc != env.wall_cartesian):
                if ((np.array(new_loc) <= (3, 3)).all() and (np.array(new_loc) >= (0, 0)).all()):
                    state = env.get_state(new_loc)
        # right (column + 1)
        elif action == Action.RIGHT:
            new_loc = (self.location[0], self.location[1] + 1)
            if (new_loc != env.wall_cartesian):
                if ((np.array(new_loc) <= (3, 3)).all() and (np.array(new_loc) >= (0, 0)).all()):
                    state = env.get_state(new_loc)
        return state


def single_episode_train(agent, env, verbosity):
    start_time = timeit.default_timer()
    # reset agent state history. V table doesn't affected by this operation.
    agent.reset(env)
    # add starting point to the history
    agent.state_history.append(env.state)
    # if verbose, draw the grid
    if verbosity >= 3:
        env.show()
    # loops until grid is solved
    steps = 0
    while not env.is_done():
        # current player makes a move
        agent.take_action(env)
        steps += 1
        # Increase epsilon as workaround to stacking in infinite actions chain
        if steps > env.grid_size * 2 and agent.epoch > 1:
            agent.epoch /= 2
    # do the value function update
    agent.update(env)
    elapsed = timeit.default_timer() - start_time
    if verbosity >= 2:
        print("Solved in %d steps" % len(agent.state_history))
        print("Time to solve grid %.3f[ms]" % (elapsed * 1000))
        print("Random actions %d, greedy actions %d" % (agent.random_actions, agent.greedy_actions))


def train_states(agent, env_factory, first, last, verbosity=0):
    steps = 0
    num_iterations = last - first + 1
    if verbosity >= 1:
        print("Train agent for %d iterations." % num_iterations)
    start_time = timeit.default_timer()
    states = set()
    for t in range(num_iterations):
        if verbosity >= 2:
            print("\nEpoch #%d" % t)
        if verbosity == 0:
            if t % int(num_iterations / 10) == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
        # For each episode create new environment so agent will face different starting positions and object locations.
        env = env_factory.create_environment(first + t)
        if env == None:
            continue
        single_episode_train(agent, env, verbosity)
        steps += len(agent.state_history)
        states.update(agent.state_history)      
        if verbosity >= 2:
            print("Episode reward: %f" % env.reward())
    
    elapsed = timeit.default_timer() - start_time
    
    if verbosity >= 2:
        print()
    if verbosity >= 1:
        print("Training time %.3f[ms]" % (elapsed * 1000))
    if verbosity == 0:
        print(" %d" % steps)
    return steps, states

def multi_agent_training(agents, env_factory, num_iterations, verbosity=0, factor=2):
    steps = 0
    env0 = env_factory.create_environment()
    if verbosity >= 1:
        print("Train agents for %d iterations." % env0.num_states)
    start_time = timeit.default_timer()
    
    if num_iterations % len(agents) != 0:
        raise RuntimeError("Number of iterations must be multiply of number of agents")
    it_per_agent = int(num_iterations / len(agents))
    
    states_per_agent = []
    for i in range(len(agents)):
        stps, states = train_states(agents[i], env_factory, i * it_per_agent, ((i + 1) * it_per_agent) - 1, verbosity)
        states_per_agent.append(np.array(list(states)))
        steps += stps
    
    if verbosity >= 1 and len(agents) > 1:
        print("Evaluate agents before mutual update")
        for agent in agents:
            evaluate_states(agent, env_factory, 0, num_iterations - 1, verbosity)
    
    # Update agents from each other
    for i in range(len(agents)):
        for j in range(len(agents)):
            # Update agent i from all agents j != i
            if i != j:
                agents[i].V[states_per_agent[j]] = (factor * agents[i].V[states_per_agent[j]] + agents[j].V[states_per_agent[j]]) / (factor + 1)

    elapsed = timeit.default_timer() - start_time
    
#     if verbosity >= 1 and len(agents) > 1:
#         print("Evaluate agents after mutual update")
#         for agent in agents:
#             evaluate_states(agent, env_factory, 0, num_iterations - 1, verbosity)
    
    if verbosity >= 1:
        print("Multi agents training time %.3f[ms]" % (elapsed * 1000))    
    return steps

def evaluate_parallel(agent, env_factory, num_iterations, verbosity=0, num_threads=1):
    if num_threads == 1:
        return evaluate(agent, env_factory, num_iterations, verbosity)
    
    if verbosity >= 1:
        print("Evaluating in parallel, each agent for %d iterations." % (num_iterations / num_threads))
        
    start_time = timeit.default_timer()
    
    args = (agent, env_factory, int(num_iterations / num_threads), verbosity)
    wg = WorkersGroup(num_threads, target=evaluate, args=args)
    res = wg.run()
    
    reward = np.array(res).mean()
    
    elapsed = timeit.default_timer() - start_time
    if verbosity >= 1:
        print("Total evaluation time %.3f[ms]" % (elapsed * 1000))
    return reward

def evaluate(agent, env_factory, num_iterations, verbosity=0):
    if verbosity >= 1:
        print("Evaluating agent for %d iterations." % num_iterations)
    start_time = timeit.default_timer()
    rewards = np.empty(num_iterations)
    for i in range(num_iterations):
        env = env_factory.create_environment()
        if verbosity >= 3:
            print()
            env.show()
        path = agent.run(env, max_steps=env.grid_size)
        
        if verbosity >= 3:
            print("Agent path")
            print(path)
            print("Reward: %.1f" % env.reward())
        rewards[i] = env.reward()
    
    elapsed = timeit.default_timer() - start_time
    if verbosity >= 1:
        print("Evaluation time %.3f[ms]" % (elapsed * 1000))
    return rewards.mean()

def evaluate_states(agent, env_factory, first, last, verbosity=0):
    if verbosity >= 2:
        print("Evaluating agent from state %d to state %d." % (first, last))
        start_time = timeit.default_timer()
    num_iterations = last - first + 1
    rewards = np.empty(num_iterations)
    rewards[:] = np.NaN
    for i in range(num_iterations):
        env = env_factory.create_environment(first + i)
        if env != None:
            if verbosity >= 3:
                print()
                env.show()
            path = agent.run(env, max_steps=env.grid_size)
            
            if verbosity >= 3 or (verbosity >= 2 and env.reward() == REWARD_PIT):
                print("Failed environment:")
                env_factory.create_environment(first + i).show()
                print("Agent path")
                print(path)
                print("Reward: %.1f" % env.reward())
            rewards[i] = env.reward()
    
    if verbosity >= 2:
        print("Valid states checked %d from total %d" % (num_iterations - len(rewards[np.isnan(rewards)]), num_iterations))
    if verbosity >= 1:
        success = rewards[rewards == REWARD_GOAL].size
        fail = rewards[rewards == REWARD_PIT].size
        hang = rewards[rewards == REWARD_HANG].size
        print("%d ended at goal, %d at pit, %d hanged." % (success, fail, hang))
    if verbosity >= 2:
        elapsed = timeit.default_timer() - start_time
        print("Evaluation time %.3f[ms]" % (elapsed * 1000))
    return np.nanmean(rewards)

np.random.seed(0)

def test_states():
    env_factory = EnvironmentFactory(EnvironmentFactory.EnvironmentType.RandomPlayerGoalAndPit)
    env0 = env_factory.create_environment()
    valid_envs = 0
    for z_player in range(env0.grid_size):
        for y_goal in range(env0.grid_size):
            for x_pit in range(env0.grid_size):
                state = int(z_player * env0.grid_size_square + y_goal * env0.grid_size + x_pit)
                env = env_factory.create_environment(state)
                if(env != None):
                    valid_envs += 1
                    try:
                        env.show()
                    except IndexError:
                        print("Error: " + str(env))
                        break 
                    
    print("Valid environments %d" % valid_envs)

def create_agent():
    agent = Agent(verbose=(verbosity >= 3))
    V = np.zeros(env.num_states)
    agent.setV(V)
    return agent
    
if __name__ == '__main__':
    # Prepare Agent
    verbosity = 1  # 0 - no verbosity; 1 - show prints between episodes; 2 - show agent log
    env_factory = EnvironmentFactory(EnvironmentFactory.EnvironmentType.RandomPlayerGoalAndPit)
    env = env_factory.create_environment()

    if verbosity >= 1:
        env.show()
        print()
    
    start_time = timeit.default_timer()
    
    if env_factory.env_type == EnvironmentFactory.EnvironmentType.RandomPlayerAndGoal:
        train_it = env.num_states
        eval_it = env.num_states * 4
    elif env_factory.env_type == EnvironmentFactory.EnvironmentType.RandomPlayerGoalAndPit:
        train_it = env.num_states
        eval_it = env.num_states * 4
    else:  
        train_it = env.num_states
        eval_it = env.num_states * 4

    converged = False
    CONVERGENCE_RATIO = 0.001
    rewards = []
    deltas = []
    prev_mean_reward = None
    total_steps = 0
    itt = 0
#     agents = deque([create_agent(), create_agent(), create_agent(), create_agent()])
#     agents = deque([create_agent(), create_agent()])
    agents = deque([create_agent()])
    factor = 2
    while not converged:
        steps = multi_agent_training(agents, env_factory, train_it, verbosity, factor=factor)
        factor *= 4
        total_steps += steps
        mean_reward = -100
        print("Evaluate agents to test convergence")
        for agent in agents:
            reward = evaluate_states(agents[0], env_factory, 0, env.num_states - 1, verbosity)
            if reward > mean_reward:
                mean_reward = reward
        rewards.append(mean_reward)
        if mean_reward == REWARD_GOAL:
            converged = True
        
#         if prev_mean_reward != None:
#             diff = mean_reward - prev_mean_reward
#             deltas.append(diff)
#             prev_mean_reward = mean_reward
#             if verbosity >= 1:
#                 print("Session %d of Training/Evaluation finished, reward: %f, delta %f." % (itt, mean_reward, diff))
#                 print("%d training steps performed in past iteration." % steps)
#             if np.abs(diff) < CONVERGENCE_RATIO:
#                 converged = True                
#         else:
#             prev_mean_reward = mean_reward
#             if verbosity >= 1:
#                 print("Session %d of Training/Evaluation finished, reward: %f." % (itt, mean_reward))
#                 print("%d training steps performed in past iteration." % steps)
        if verbosity >= 1:
            print()
        itt += 1        
        agents.rotate()
    
    elapsed = timeit.default_timer() - start_time
    
    if verbosity >= 1:
        for agent in agents:
            evaluate_states(agent, env_factory, 0, env.num_states - 1, verbosity=2)
    
    print()
    print("Training finished after %d iterations. Total learning steps are %d." % (itt, total_steps))
    print("Total time spent %.3f[s]" % elapsed)
    print("Final reward %f" % rewards[-1:][0])
    print("Saving V table to vtable.bin")
#     V.tofile('vtable.bin')
    plt.plot(rewards, 'g')
    plt.show()
    print('Done')
