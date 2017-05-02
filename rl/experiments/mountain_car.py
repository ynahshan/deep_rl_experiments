import gym
import numpy as np
import pandas as pd
import os

from gym import wrappers
from datetime import datetime

from rl.models.linear_models import RbfRegressor
from rl.agents.qlearning_agent import QLearningFunctionAproximationAgent

import matplotlib.pyplot as plt

def create_model(env, verbose=False):
    obs = np.array([env.observation_space.sample()])
    obs_dim = obs.ndim
    observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
    model = RbfRegressor(in_size=obs_dim, num_features=1000, output_size=env.action_space.n, gammmas=[5.0, 2.0, 1.0, 0.5], verbose=verbose)
    model.fit_features(observation_examples, env)
    return model

class EnvDescriptor(object):
    def __init__(self):
        self.episod_limit = 2000
    def action_to_str(self, action):
        return str(action)

np.random.seed(0)
if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env.seed(0)
    verbose = False
    model = create_model(env, verbose=verbose)
    gamma = 0.99
    agent = QLearningFunctionAproximationAgent(model=model, eps=0.2, gamma=gamma, env_descriptor=EnvDescriptor(), verbose=verbose)

    monitor = True
    if monitor:
        filename = os.path.basename(__file__).split('.')[0]
        monitor_dir = filename + '_' + str(datetime.now()).replace(' ', '_').replace(':', '_')
        monitor_dir = os.path.join(monitor_dir, os.pardir, os.pardir, os.pardir, 'temp')
        env = wrappers.Monitor(env, monitor_dir, force=True)

    num_iter = 280
    total_steps = 0
    returns = []
    for i in range(num_iter):
        print("Epoch %d" % i)
        steps, tot_ret, last_reward = agent.single_episode_train(env)
        total_steps += steps
        returns.append(tot_ret)
        # if i > 0 and (i % 50 == 0):
        #     model.adjust()
        print('Epoch finished in %d steps. Return %f.' % (steps, tot_ret))

    env.close()

    cum_returns = (pd.DataFrame(returns, columns=['r'])).r.rolling(window=100).mean()

    print("Last 100 episode average reward %f" % np.mean(returns[-100:]))

    fig, ax = plt.subplots()
    ax.plot(returns, label='Returns')
    ax.plot(cum_returns, label='Cumulative return')

    plt.show()
    print('Done')