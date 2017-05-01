from rl.environments import gym_like

from rl.environments.basic_gird_world import BasicGridWorld_v0 as bgw
gym_like.register(bgw.name, bgw)

from rl.environments.basic_gird_world import BasicGridWorld_v1 as bgw
gym_like.register(bgw.name, bgw)