import numpy as np
import matplotlib.pyplot as plt
from math import gamma

np.random.seed(0)

class Bandit:
    def __init__(self, m):
        self.m = m
        self.mean = 0
        self.N = 0

    def pull(self):
        return np.random.randn() + self.m

    def update(self, x):
        self.N += 1
        self.mean = (1 - 1.0/self.N)*self.mean + 1.0/self.N*x

def run_experiment_eps_greedy(m1, m2, m3, eps, N, decay_interval, gamma=0.9):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    data = np.empty(N)
    alpha=1
    for i in range(N):
        if i % decay_interval == 0:
            alpha*=gamma
        # epsilon greedy
        p = np.random.random()
        if p < eps*alpha:
            j = np.random.choice(3)
        else:
            j = np.argmax([b.mean for b in bandits])
        x = bandits[j].pull()
        bandits[j].update(x)
        
        # for the plot
        data[i] = x
    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)
    
    # plot moving average ctr
#     plt.plot(cumulative_average)
#     plt.plot(np.ones(N)*m1)
#     plt.plot(np.ones(N)*m2)
#     plt.plot(np.ones(N)*m3)
#     plt.xscale('log')
#     plt.show()
    
    for b in bandits:
        print (b.mean)
    
    return cumulative_average

def run_experiment(N, decay_interval, gamma):
    c_1 = run_experiment_eps_greedy(1.0, 2.0, 3.0, 0.1, N, decay_interval, gamma)
    c_2 = run_experiment_eps_greedy(1.0, 2.0, 3.0, 0.2, N, decay_interval, gamma)
    c_3 = run_experiment_eps_greedy(1.0, 2.0, 3.0, 0.5, N, decay_interval, gamma)
    
    # log scale plot
    plt.plot(np.ones(N)*3, 'y')
#     rng=range(1000,N)
    plt.plot(c_1, label='eps = 0.1')
    plt.plot(c_2, label='eps = 0.2')
    plt.plot(c_3, label='eps = 0.5')
    plt.legend()
    plt.xscale('log')
    plt.show()
    
if __name__ == '__main__':
    N=10000
    run_experiment(N, 100, 0.7)
    run_experiment(N, 10, 0.95)
    
    print('Done')
