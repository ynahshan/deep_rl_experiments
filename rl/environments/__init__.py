from rl.environments import gym_like

from rl.environments.basic_gird_world import BasicGridWorld
gym_like.register(BasicGridWorld.name, BasicGridWorld)