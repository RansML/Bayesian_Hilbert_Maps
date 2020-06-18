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
device = pt.device("cpu")
#device = pt.device("cuda:0") # Uncomment this to run on GPU

#TODO: merege 2D and 3D classes into a single class
#TODO: get rid of all numpy operations and test on a GPU
#TODO: parallelizing the segmentations
#TODO: efficient querying
#TODO: batch training
#TODO: re-using parameters for moving vehicles
class BHM2D_PYTORCH():
    def __init__(self, gamma=0.05, grid=None, cell_resolution=(5, 5), cell_max_min=None, X=None, nIter=0, mu_sig=None):
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

        #ADDED
        if mu_sig is not None:
            self.mu = pt.tensor(mu_sig[:,0], dtype=pt.float32)
            self.sig = pt.tensor(mu_sig[:,1], dtype=pt.float32)

    def updateGrid(grid):
        self.grid = grid

    def updateMuSig(mu_sig):
        self.mu = pt.tensor(mu_sig[:,0], dtype=pt.float32)
        self.sig = pt.tensor(mu_sig[:,1], dtype=pt.float32)

    def __calc_grid_auto(self, cell_resolution, max_min, X):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """
        # X = X.numpy()

        if max_min is None:
            # if 'max_min' is not given, make a boundarary based on X
            # assume 'X' contains samples from the entire area
            expansion_coef = 1.2
            x_min, x_max = expansion_coef*X[:, 0].min(), expansion_coef*X[:, 0].max()
            y_min, y_max = expansion_coef*X[:, 1].min(), expansion_coef*X[:, 1].max()
        else:
            x_min, x_max = max_min[0], max_min[1]
            y_min, y_max = max_min[2], max_min[3]

        xx, yy = np.meshgrid(np.arange(x_min, x_max, cell_resolution[0]), \
                             np.arange(y_min, y_max, cell_resolution[1]))
        grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))

        return pt.tensor(grid)

    def __sparse_features(self, X):
        """
        :param X: inputs of size (N,2)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        rbf_features = rbf_kernel(X, self.grid, gamma=self.gamma)
        # COMMENTED OUT BIAS TERM
        # rbf_features = np.hstack((np.ones(X.shape[0])[:, np.newaxis], rbf_features))
        return pt.tensor(rbf_features, dtype=pt.float32)

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

        self.epsilon = pt.ones(N, dtype=pt.float32)
        if not hasattr(self, 'mu'):
            self.mu = pt.zeros(D, dtype=pt.float32)
            self.sig = 10000 * pt.ones(D, dtype=pt.float32)

        for i in range(self.nIter):
            print("  Parameter estimation: iter={}".format(i))

            # E-step
            self.mu, self.sig = self.__calc_posterior(X, y, self.epsilon, self.mu, self.sig)

            # M-step
            self.epsilon = pt.sqrt(pt.sum((X**2)*self.sig, dim=1) + (X.mm(self.mu.reshape(-1, 1))**2).squeeze())

        # print(self.mu)

        return self.mu, self.sig


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

        mean = pt.mean(probs, dim=1).squeeze()
        std = pt.std(probs, dim=1).squeeze()

        return mean, std


class BHM3D_PYTORCH():
    def __init__(self, gamma=0.05, grid=None, cell_resolution=(5, 5), cell_max_min=None, X=None, nIter=2, mu_sig=None):
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

    def updateGrid(grid):
        self.grid = grid

    def updateMuSig(mu_sig):
        self.mu = pt.tensor(mu_sig[:,0], dtype=pt.float32)
        self.sig = pt.tensor(mu_sig[:,1], dtype=pt.float32)

    def __calc_grid_auto(self, cell_resolution, max_min, X):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max, z_min, z_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """
        X = X.numpy()

        if max_min is None:
            # if 'max_min' is not given, make a boundarary based on X
            # assume 'X' contains samples from the entire area
            expansion_coef = 1.2
            x_min, x_max = expansion_coef*X[:, 0].min(), expansion_coef*X[:, 0].max()
            y_min, y_max = expansion_coef*X[:, 1].min(), expansion_coef*X[:, 1].max()
        else:
            x_min, x_max = max_min[0], max_min[1]
            y_min, y_max = max_min[2], max_min[3]
            z_min, z_max = max_min[4], max_min[5]

        xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, cell_resolution[0]), \
                             np.arange(y_min, y_max, cell_resolution[1]), \
                             np.arange(z_min, z_max, cell_resolution[2]))
        grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis], zz.ravel()[:, np.newaxis]))

        return pt.tensor(grid)

    def __sparse_features(self, X):
        """
        :param X: inputs of size (N,3)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        rbf_features = rbf_kernel(X, self.grid, gamma=self.gamma)

        # rbf_features = np.hstack((np.ones(X.shape[0])[:, np.newaxis], rbf_features))
        return pt.tensor(rbf_features, dtype=pt.float32)

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

        self.epsilon = pt.ones(N, dtype=pt.float32)
        if not hasattr(self, 'mu'):
            self.mu = pt.zeros(D, dtype=pt.float32)
            self.sig = 10000 * pt.ones(D, dtype=pt.float32)

        for i in range(self.nIter):
            print("  Parameter estimation: iter={}".format(i))

            # E-step
            self.mu, self.sig = self.__calc_posterior(X, y, self.epsilon, self.mu, self.sig)

            # M-step
            self.epsilon = pt.sqrt(pt.sum((X**2)*self.sig, dim=1) + (X.mm(self.mu.reshape(-1, 1))**2).squeeze())
        return self.mu, self.sig

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

        mean = pt.mean(probs, dim=1).squeeze()
        std = pt.std(probs, dim=1).squeeze()

        return mean, std


class BHM_FULL_PYTORCH():
    #TODO: double check evrything
    def __init__(self, gamma=0.075*0.814, grid=None, cell_resolution=(5, 5), cell_max_min=None, X=None):
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
        print('D=', self.grid.shape[0])

    def __calc_grid_auto(self, cell_resolution, max_min, X):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """
        X = X.numpy()

        if max_min is None:
            # if 'max_min' is not given, make a boundarary based on X
            # assume 'X' contains samples from the entire area
            expansion_coef = 1.2
            x_min, x_max = expansion_coef*X[:, 0].min(), expansion_coef*X[:, 0].max()
            y_min, y_max = expansion_coef*X[:, 1].min(), expansion_coef*X[:, 1].max()
        else:
            x_min, x_max = max_min[0], max_min[1]
            y_min, y_max = max_min[2], max_min[3]

        xx, yy = np.meshgrid(np.arange(x_min, x_max, cell_resolution[0]), \
                             np.arange(y_min, y_max, cell_resolution[1]))
        grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))

        return pt.tensor(grid)

    def __sparse_features(self, X):
        """
        :param X: inputs of size (N,2)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        rbf_features = rbf_kernel(X, self.grid, gamma=self.gamma)
        rbf_features = np.hstack((np.ones(X.shape[0])[:, np.newaxis], rbf_features))
        return pt.tensor(rbf_features, dtype=pt.float32)

    def __lambda(self, epsilon):
        logit_inv = pt.sigmoid(epsilon)
        return 0.5 / epsilon * (logit_inv - 0.5)

    def __calc_posterior(self, X, y, epsilon, mu0, sig0_inv):
        lam = self.__lambda(epsilon)
        mu0 = mu0.reshape(-1,1)

        #sig = 1/(1/sig0 + 2*pt.sum( (X.t()**2)*lam, dim=1))
        sig_inv = sig0_inv + 2*pt.mm(X.t() * lam, X)
        sig = pt.inverse(sig_inv)
        #sig = pt.diag(1/pt.diag(sig_inv))

        print("sum=", pt.sum(sig_inv.mm(sig)))

        pl.subplot(121)
        pl.imshow(sig_inv[1:,1:], cmap='jet', interpolation=None); pl.colorbar()
        pl.subplot(122)
        pl.imshow(sig[1:,1:], cmap='jet', interpolation=None); pl.colorbar()
        pl.show()

        #mu = sig*(mu0/sig0 + pt.mm(X.t(), y - 0.5).squeeze())
        #sig_inv = sig0_inv + 2 * pt.diag(pt.mm(X.t() * lam, X)).squeeze()
        #mu = pt.mm((1 / sig_inv).reshape(1, -1), (sig_inv * mu0 + pt.mm(X.t(), y - 0.5))).squeeze()
        mu = sig.mm(sig0_inv.mm(mu0) + pt.mm(X.t(), (y - 0.5)))

        pl.close('all')
        pl.scatter(self.grid[:,0], self.grid[:,1], c=mu.squeeze()[1:], cmap='jet', s=5, edgecolor='')
        pl.colorbar()
        pl.show()

        return mu.squeeze(), sig_inv, sig

    def fit(self, X, y):
        X = self.__sparse_features(X)
        N, D = X.shape[0], X.shape[1]

        epsilon = pt.ones(N, dtype=pt.float32)
        mu = pt.zeros(D, dtype=pt.float32)
        sig_inv = 0.0001 * pt.eye(D, dtype=pt.float32)

        for i in range(1):
            print("i=", i)

            # E-step
            mu, sig_inv, sig = self.__calc_posterior(X, y, epsilon, mu, sig_inv)
            print('d', mu.shape)

            # M-step
            XMX = pt.mv(X, mu)**2
            XSX = pt.sum(pt.mm(X, pt.mm(sig, X.t())), dim=1)
            print(XMX.shape, XSX.shape)
            epsilon = pt.sqrt(XMX + XSX) #TODO - Bug

        self.mu, self.sig_inv = mu, sig_inv

    def predict(self, Xq):
        Xq = self.__sparse_features(Xq)

        sig_diag = 1/pt.diagonal(self.sig_inv)

        mu_a = Xq.mm(self.mu.reshape(-1, 1)).squeeze()
        sig2_inv_a = pt.sum((Xq ** 2) * sig_diag, dim=1)
        k = 1.0 / pt.sqrt(1 + np.pi * sig2_inv_a / 8)

        return pt.sigmoid(k*mu_a)
