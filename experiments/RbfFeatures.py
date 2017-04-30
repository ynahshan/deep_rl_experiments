'''
Created on Apr 29, 2017

@author: Yury
'''

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import FeatureUnion

np.random.seed(0)
if __name__ == '__main__':
    X = np.arange(16).astype(float)
    print("Input data")
    print(X)
    X = X.reshape(X.size, 1)
    scaler = StandardScaler()
    scaler.fit(X)
    X_norm = scaler.transform(X)
    print("Normalized input data")
    print(X_norm.squeeze())
    print()
    
    n_cmp = 4
#     rbfs = RBFSampler(gamma=1, n_components=n_cmp)
    rbfs  = FeatureUnion([
                ("rbf1", RBFSampler(gamma=0.05, n_components=n_cmp)),
                ("rbf2", RBFSampler(gamma=0.1, n_components=n_cmp)),
                ("rbf3", RBFSampler(gamma=0.5, n_components=n_cmp)),
                ("rbf4", RBFSampler(gamma=1.0, n_components=n_cmp)),
                ("rbf5", RBFSampler(gamma=1.5, n_components=n_cmp)),
                ("rbf6", RBFSampler(gamma=2.0, n_components=n_cmp))
                ])
    X_features = rbfs.fit_transform(X)
    
    print("Dimensions of feature space is %d\n" % X_features.shape[1])
    
    print("RBF Features of input data")
    print(X_features.squeeze())
    print()
    
    Y = np.array([3, 3, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 3, 3, 1])
    reg = SGDRegressor(n_iter=100)
    reg.fit(X_features, Y)
    print("Regressor coefficients")
    print(reg.coef_)
    print()
    
    x = 0
    print("Try to predict Y")
    print(Y.squeeze())
#     print("Prediction of %d is" % x)
    p = reg.predict(rbfs.transform(X))
    print(p)
    
    print("Diff")
    print(Y - p)
    print(np.abs(Y - p).mean())
    print('Done')
    pass
