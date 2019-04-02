"""
# 2D and 3D Bayesian Hilbert Maps with pytorch
# Ransalu Senanayake
"""
import torch as pt
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as pl
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
import pandas as pd

dtype = pt.float32
#device = pt.device("cpu")
device = pt.device("cuda:0") # Uncomment this to run on GPU

#TODO: merege 2D and 3D classes into a single class
#TODO: get rid of all numpy operations and test on a GPU
#TODO: parallelizing the segmentations
#TODO: efficient querying
#TODO: batch training
#TODO: re-using parameters for moving vehicles

class BHM2D_PYTORCH_CUDA():
    def __init__(self, gamma=0.05, grid=None, cell_resolution=(5, 5), cell_max_min=None, X=None, nIter=0):
        """
        :param gamma: RBF bandwidth
        :param grid: if there are prespecified locations to hinge the RBF
        :param cell_resolution: if 'grid' is 'None', resolution to hinge RBFs
        :param cell_max_min: if 'grid' is 'None', realm of the RBF field
        :param X: a sample of lidar locations to use when both 'grid' and 'cell_max_min' are 'None'
        """
        self.gamma = gamma
        if grid is not None:
            self.grid = grid
        else:
            self.grid = self.__calc_grid_auto(cell_resolution, cell_max_min, X)
        self.nIter = nIter
        print(' Number of hinge points={}'.format(self.grid.shape[0]))

    def __calc_grid_auto(self, cell_resolution, max_min, X):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """
        #X = X.numpy()

        if max_min is None:
            # if 'max_min' is not given, make a boundarary based on X
            # assume 'X' contains samples from the entire area
            expansion_coef = 1.2
            x_min, x_max = expansion_coef*X[:, 0].min(), expansion_coef*X[:, 0].max()
            y_min, y_max = expansion_coef*X[:, 1].min(), expansion_coef*X[:, 1].max()
        else:
            x_min, x_max = max_min[0], max_min[1]
            y_min, y_max = max_min[2], max_min[3]

        xx, yy = pt.meshgrid([pt.arange(x_min, x_max, cell_resolution[0]).cuda(),\
                              pt.arange(y_min, y_max, cell_resolution[1]).cuda()])
        grid = pt.stack((xx.reshape(-1,1), yy.reshape(-1,1)), dim=1).squeeze()

        return grid

    def __rbf_kernel(self, X1, X2, gamma):

        K = pt.norm(X1[:, None] - X2, dim=-1, p=2).pow(2)
        K = pt.exp(-gamma*K)

        return K

    def __sparse_features(self, X):
        """
        :param X: inputs of size (N,2)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        rbf_features = self.__rbf_kernel(X, self.grid, gamma=self.gamma)
        rbf_features = pt.cat((pt.ones(X.shape[0],1).cuda(), rbf_features), dim=1)

        return rbf_features

    def __calc_posterior(self, X, y, epsilon, mu0, sig0):
        """
        :param X: input features
        :param y: labels
        :param epsilon: per dimension local linear parameter
        :param mu0: mean
        :param sig0: variance
        :return: new_mean, new_varaiance
        """
        logit_inv = pt.sigmoid(epsilon)
        lam = 0.5 / epsilon * (logit_inv - 0.5)
        sig = 1/(1/sig0 + 2*pt.sum( (X.t()**2)*lam, dim=1))
        mu = sig*(mu0/sig0 + pt.mm(X.t(), y - 0.5).squeeze())
        return mu, sig

    def fit(self, X, y):
        """
        :param X: raw data
        :param y: labels
        """
        X = self.__sparse_features(X)
        N, D = X.shape[0], X.shape[1]

        self.epsilon = pt.ones(N, dtype=pt.float32).cuda()
        if not hasattr(self, 'mu'):
            self.mu = pt.zeros(D, dtype=pt.float32).cuda()
            self.sig = 10000 * pt.ones(D, dtype=pt.float32).cuda()

        for i in range(self.nIter):
            print("  Parameter estimation: iter={}".format(i))

            # E-step
            self.mu, self.sig = self.__calc_posterior(X, y, self.epsilon, self.mu, self.sig)

            # M-step
            self.epsilon = pt.sqrt(pt.sum((X**2)*self.sig, dim=1) + (X.mm(self.mu.reshape(-1, 1))**2).squeeze())

    def predict(self, Xq):
        """
        :param Xq: raw inquery points
        :return: mean occupancy (Lapalce approximation)
        """
        Xq = self.__sparse_features(Xq)

        mu_a = Xq.mm(self.mu.reshape(-1, 1)).squeeze()
        sig2_inv_a = pt.sum((Xq ** 2) * self.sig, dim=1)
        k = 1.0 / pt.sqrt(1 + np.pi * sig2_inv_a / 8)

        return pt.sigmoid(k*mu_a)

    def predictSampling(self, Xq, nSamples=50):
        """
        :param Xq: raw inquery points
        :param nSamples: number of samples to take the average over
        :return: sample mean and standard deviation of occupancy
        """
        Xq = self.__sparse_features(Xq)

        qw = pt.distributions.MultivariateNormal(self.mu, pt.diag(self.sig))
        w = qw.sample((nSamples,)).t()

        mu_a = Xq.mm(w).squeeze()
        probs = pt.sigmoid(mu_a)

        mean = pt.std(probs, dim=1).squeeze()
        std = pt.std(probs, dim=1).squeeze()

        return mean, std