'''
Created on Mar 13, 2017

@author: Yury
'''

import numpy as np
import timeit
import matplotlib.pyplot as plt
import sys
from multiprocessing.pool import ThreadPool

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

class EnvironmentFactory:
    class EnvironmentType:
        Deterministic = 0
        RandomPlayer = 1
        RandomPlayerAndGoal = 2
        
    def __init__(self, env_type):
        self.env_type = env_type
        
    def create_environment(self): 
        if self.env_type == EnvironmentFactory.EnvironmentType.RandomPlayer:
            env = DeterministicEnvironment()
            start_pos = np.random.choice(env.num_states)
            while start_pos in [env.goal, env.pit, env.wall]:
                start_pos = np.random.choice(env.num_states)
            env.player_starting_point = start_pos
            env.state = start_pos
        elif self.env_type == EnvironmentFactory.EnvironmentType.RandomPlayerAndGoal:
            env = RandomGoalAndPlayerEnvironment()
        else:
            env = DeterministicEnvironment()
            
        return env
            
class RandomGoalAndPlayerEnvironment():
    def __init__(self):
        self.size = 4
        self.grid_size = self.size * self.size
        self.num_states = self.grid_size ** 2
        self.wall = 10
        self.wall_cartesian = (2, 2)
        self.pit = 5
        self.pit_cartesian = (1, 1)        
        self.goal = np.random.choice(self.grid_size)
        while self.goal in [self.wall, self.pit]:
            self.goal = np.random.choice(self.grid_size)
        self.goal_cartesian = (int(self.goal / self.size), int(self.goal % self.size))
        self.player_starting_point = np.random.choice(self.grid_size)
        while self.player_starting_point in [self.wall, self.pit, self.goal]:
            self.player_starting_point = np.random.choice(self.grid_size)
        self.state = int(self.player_starting_point * self.grid_size) + self.goal
        
    def update_state(self, player_loc):
        player_pos = player_loc[0] * self.size + player_loc[1]
        self.state = int(player_pos * self.grid_size) + self.goal
        
    def get_state(self, player_loc):
        player_pos = player_loc[0] * self.size + player_loc[1]
        return int(player_pos * self.grid_size) + self.goal
    
    def is_done(self):
        player_pos = int(self.state / self.grid_size)
        return player_pos == self.pit or player_pos == self.goal
    
    def reward(self):
        player_pos = int(self.state / self.grid_size)
        if player_pos == self.pit:
            return -10
        elif player_pos == self.goal:
            return 10
        else:
            return -10
        
    def show(self):
        grid = np.zeros((env.size, env.size), dtype='<U2')
        player_pos = int(self.state / self.grid_size)
        player_loc = (int(player_pos / self.size), player_pos % self.size)

        for i in range(0, env.size):
            for j in range(0, env.size):
                grid[i, j] = ' '
    

        grid[player_loc] = 'P'  # player starting point
        grid[self.wall_cartesian] = 'W'  # wall
        grid[self.goal_cartesian] = '+'  # goal
        grid[self.pit_cartesian] = '-'  # pit
    
        print(grid)
    
class DeterministicEnvironment:
    def __init__(self):
        self.size = 4
        self.grid_size = self.size * self.size
        self.num_states = self.grid_size
        self.player_starting_point = 0
        self.state = self.player_starting_point
        self.wall = 10
        self.wall_cartesian = (2, 2)
        self.goal = 15
        self.goal_cartesian = (3, 3)
        self.pit = 5
        self.pit_cartesian = (1, 1)
    
    def update_state(self, player_loc):
        self.state = int(player_loc[0] * self.size + player_loc[1])
        
    def get_state(self, player_loc):
        return player_loc[0] * self.size + player_loc[1]
    
    def is_done(self):
        return self.state == self.pit or self.state == self.goal 
    
    def reward(self):
        if self.state == self.pit:
            return -10
        elif self.state == self.goal:
            return 10
        else:
            return -1
        
    def show(self):
        grid = np.zeros((env.size, env.size), dtype='<U2')
        player_loc = (int(self.state / self.size), self.state % self.size)
        wall = (int(self.wall / self.size), self.wall % self.size)
        goal = (int(self.goal / self.size), self.goal % self.size)
        pit = (int(self.pit / self.size), self.pit % self.size)

        for i in range(0, env.size):
            for j in range(0, env.size):
                grid[i, j] = ' '
    
        if player_loc:
            grid[player_loc] = 'P'  # player starting point
        if wall:
            grid[wall] = 'W'  # wall
        if goal:
            grid[goal] = '+'  # goal
        if pit:
            grid[pit] = '-'  # pit
    
        print(grid)
    
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
        V[self.state_history.pop()] = target
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


def train(agent, env_factory, num_iterations, verbosity=0):
    steps = 0
    if verbosity >= 1:
        print("Train agent for %d iterations." % num_iterations)
    start_time = timeit.default_timer()
    for t in range(num_iterations):
        if verbosity >= 2:
            print("\nEpoch #%d" % t)
        if verbosity == 0:
            if t % int(num_iterations / 10) == 0:
                sys.stdout.write('.')
                sys.stdout.flush()
        # For each episode create new environment so agent will face different starting positions and object locations.
        env = env_factory.create_environment()
        single_episode_train(agent, env, verbosity)
        steps += len(agent.state_history)      
        if verbosity >= 2:
            print("Episode reward: %f" % env.reward())
    
    elapsed = timeit.default_timer() - start_time
    
    if verbosity >= 2:
        print()
    if verbosity >= 1:
        print("Training time %.3f[ms]" % (elapsed * 1000))
    if verbosity == 0:
        print(" %d" % steps)
    return steps

def evaluate_parallel(agent, env_factory, num_iterations, verbosity=0, num_threads=1):
    if num_threads == 1:
        return evaluate(agent, env_factory, num_iterations, verbosity)
    
    if verbosity >= 1:
        print("Evaluating in parallel, each agent for %d iterations." % (num_iterations / num_threads))
        
    start_time = timeit.default_timer()
    
#     agents = [agent]
#     for _ in range(num_threads - 1):
#         agents.append(agent.copy())
    
    pool = ThreadPool(num_threads)
    res = []
    for _ in range(num_threads):
        r = pool.apply_async(evaluate, (agent, env_factory, int(num_iterations / num_threads), verbosity))
        res.append(r)
    
    rewards = 0
    for r in res:
        rewards += r.get()
    
    elapsed = timeit.default_timer() - start_time
    if verbosity >= 1:
        print("Total evaluation time %.3f[ms]" % (elapsed * 1000))
    return rewards / num_threads

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

np.random.seed(0)

if __name__ == '__main__':
    # Prepare Agent
    verbosity = 0  # 0 - no verbosity; 1 - show prints between episodes; 2 - show agent log
    env_factory = EnvironmentFactory(EnvironmentFactory.EnvironmentType.RandomPlayerAndGoal)
    env = env_factory.create_environment()
    agent = Agent(verbose=(verbosity >= 3))
    if verbosity >= 1:
        env.show()
        print()
    V = np.zeros(env.num_states)
    agent.setV(V)
    
    start_time = timeit.default_timer()
    
    train_it = 20 if env_factory.env_type == EnvironmentFactory.EnvironmentType.RandomPlayerAndGoal else 10
    eval_it = env.num_states * 4
    converged = False
    CONVERGENCE_RATIO = 0.001
    rewards = []
    deltas = []
    prev_mean_reward = None
    total_steps = 0
    itt = 0
    while not converged:
        steps = train(agent, env_factory, train_it, verbosity)
        total_steps += steps
        mean_reward = evaluate_parallel(agent, env_factory, eval_it, verbosity, num_threads=4)
        rewards.append(mean_reward)
        if prev_mean_reward != None:
            diff = mean_reward - prev_mean_reward
            deltas.append(diff)
            prev_mean_reward = mean_reward
            if verbosity >= 1:
                print("Session %d of Training/Evaluation finished, reward: %f, delta %f." % (itt, mean_reward, diff))
                print("%d training steps performed in past iteration." % steps)
            if np.abs(diff) < CONVERGENCE_RATIO:
                converged = True                
        else:
            prev_mean_reward = mean_reward
            if verbosity >= 1:
                print("Session %d of Training/Evaluation finished, reward: %f." % (itt, mean_reward))
                print("%d training steps performed in past iteration." % steps)
        if verbosity >= 1:
            print()
        itt += 1        
    
    elapsed = timeit.default_timer() - start_time
    
    print()
    print("Training finished after %d iterations. Total learning steps are %d." % (itt, total_steps))
    print("Total time spent %.3f[s]" % elapsed)
    print("Final reward %f" % rewards[-1:][0])
    print("Saving V table to vtable.bin")
    V.tofile('vtable.bin')
    plt.plot(rewards, 'g', deltas, 'r')
    plt.show()
    print('Done')



