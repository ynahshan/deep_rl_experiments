'''
Created on Apr 30, 2017

@author: ny
'''

from rl.environments import gym_like as gym

if __name__ == '__main__':
    print(gym.env_list())
    
    env = gym.make('BasicGridWorld-v0')
    print(env.reset())
    
    env.render()
    
    try:
        env_name = 'Some_not_existing_environment'
        env = gym.make(env_name)
    except:
        print("%s doesn't exist as expected" % env_name)
    
    pass