import gym
import numpy as np
import pandas as pd
import os

from gym import wrappers
from datetime import datetime

from rl_gym.models.linear_models import RbfRegressor
from rl_gym.models.mlp_models import FeedForwardModel
from rl_gym.agents.qlearning_agent import QLearningFunctionAproximationAgent

import matplotlib.pyplot as plt

def create_model(env, verbose=False):
    obs = np.array([env.observation_space.sample()])
    obs_dim = obs.ndim
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    model = RbfRegressor(in_size=obs_dim, num_features=1000, output_size=env.action_space.n, gammmas=[5.0, 2.0, 1.0, 0.5], verbose=verbose)
    model.fit_features(observation_examples, env)
    return model

def create_model(env, model_name, verbose=False):
    obs = env.reset()
    obs_dim = len(obs)
    if model_name == 'rbf':
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        model = RbfRegressor(in_size=obs_dim, num_features=500, output_size=env.action_space.n, gammmas=[5.0, 2.0, 1.0, 0.5], verbose=verbose)
        model.fit_features(observation_examples, env)
        gamma = 0.99
    elif model_name == 'ff':
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        model = FeedForwardModel(in_size=obs_dim, out_sizes=[128, 64, 32, env.action_space.n], normalize=False, verbose=verbose)
        model.fit_features(observation_examples)
        gamma = 0.99

    return model, gamma

def set_monitor(env):
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = filename + '_' + str(datetime.now()).replace(' ', '_').replace(':', '_')
    monitor_dir = os.path.join(monitor_dir, os.pardir, os.pardir, os.pardir, 'temp')
    env = wrappers.Monitor(env, monitor_dir, force=True)

np.random.seed(0)
if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env.seed(0)
    verbose = False
    models = ['rbf', 'ff']
    model, gamma = create_model(env, models[0], verbose=verbose)
    agent = QLearningFunctionAproximationAgent(model=model, eps=0.0, eps_decay=0.99, gamma=gamma, verbose=verbose)

    monitor = True
    if monitor:
        set_monitor(env)

    num_iter = 40
    total_steps = 0
    returns = []
    for i in range(num_iter):
        print("Epoch %d" % i)
        # if i == 23:
        #     agent.verbose = True
        # else:
        #     agent.verbose = False
        steps, tot_ret, last_reward = agent.single_episode_train(env)
        total_steps += steps
        returns.append(tot_ret)
        # if i > 0 and (i % 50 == 0):
        #     model.adjust()
        print('Episode finished in %d steps. Return %f.' % (steps, tot_ret))

    env.close()

    cum_returns = (pd.DataFrame(returns, columns=['r'])).r.rolling(window=100).mean()

    print("Last 100 episode average reward %f" % np.mean(returns[-100:]))

    fig, ax = plt.subplots()
    ax.plot(returns, label='Returns')
    ax.plot(cum_returns, label='Cumulative return')

    plt.show()
    print('Done')