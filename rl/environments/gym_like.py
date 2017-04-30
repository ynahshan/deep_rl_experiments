'''
Created on Apr 30, 2017

@author: ny
'''

registry = {}

def env_list():
    return list(registry.keys())

def register(env_name, env_cls):
    registry[env_name] = env_cls

def make(env_name):
    if env_name in registry:
        return registry[env_name]()
    else:
        raise RuntimeError("Environment with name %s not registered.")