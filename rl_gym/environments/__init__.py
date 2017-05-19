from rl_gym.environments import gym_like

from rl_gym.environments.basic_gird_world import BasicGridWorld_v0 as bgw
gym_like.register(bgw.name, bgw)

from rl_gym.environments.basic_gird_world import BasicGridWorld_v1 as bgw
gym_like.register(bgw.name, bgw)