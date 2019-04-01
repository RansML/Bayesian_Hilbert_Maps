import numpy as np
import copy
import matplotlib.pylab as pl
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
import sys
#import util
import time as comp_timer
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
from scipy.special import expit
from scipy.linalg import solve_triangular

class SBHM(LinearClassifierMixin, BaseEstimator):
    def __init__(self, gamma=0.075*0.814, grid=None, cell_resolution=(5, 5), cell_max_min=None, X=None, calc_loss=False):
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
        self.calc_loss = calc_loss
        self.intercept_, self.coef_, self.sigma_ = [0], [0], [0]
        self.scan_no = 0
        print('D=', self.grid.shape[0])

    def __calc_grid_auto(self, cell_resolution, max_min, X):
        """
        :param X: a sample of lidar locations
        :param cell_resolution: resolution to hinge RBFs as (x_resolution, y_resolution)
        :param max_min: realm of the RBF field as (x_min, x_max, y_min, y_max)
        :return: numpy array of size (# of RNFs, 2) with grid locations
        """

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

        return grid

    def __sparse_features(self, X):
        """
        :param X: inputs of size (N,2)
        :return: hinged features with intercept of size (N, # of features + 1)
        """
        rbf_features = rbf_kernel(X, self.grid, gamma=self.gamma)
        return np.hstack((np.ones(X.shape[0])[:, np.newaxis], rbf_features))

    def __lambda(self, epsilon):
        """
        :param epsilon: epsilon value for each data point
        :return: local approximation
        """
        return 0.5/epsilon*(expit(epsilon)-0.5)

    def __calc_loss(self, X, mu0, Sig0_inv, mu, Sig_inv, Ri, epsilon):
        # TODO
        Sig = np.dot(Ri.T, Ri)
        R0 = np.linalg.cholesky(Sig0_inv)
        R0i = solve_triangular(R0, np.eye(X.shape[1]), lower=True)
        S0 = np.dot(R0i.T, R0i)
        loss = 0.5 * np.linalg.slogdet(Sig)[1] - 0.5 * np.linalg.slogdet(S0)[1] + 0.5 * mu.T.dot(Sig_inv.dot(mu)) - 0.5 * mu0.T.dot(Sig0_inv.dot(mu0))
        loss += (np.sum(np.log(expit(epsilon)) - 0.5 * epsilon + self.__lambda(epsilon) * epsilon ** 2))

        return loss

    def __calc_posterior(self, X, y, epsilon, mu0, Sig0_inv, full_covar=False):
        """
        :param X: positions
        :param y: labels
        :param epsilon:
        :param mu0: mean
        :param Sig0_inv: inverse of covariance
        :param full_covar: whether to output the full covariance or not
        :return: mean, inverse of covariance, Cholesky factor
        """

        lam = self.__lambda(epsilon)

        Sig_inv = Sig0_inv + 2 * np.dot(X.T*lam, X)

        m_right = Sig0_inv.dot(mu0) + np.dot(X.T, (y - 0.5))
        L_of_Sig_inv = np.linalg.cholesky(Sig_inv)
        Z = solve_triangular(L_of_Sig_inv, m_right, lower=True)
        mu = solve_triangular(L_of_Sig_inv.T, Z, lower=False)

        L_inv_of_Sig_inv = solve_triangular(L_of_Sig_inv, np.eye(X.shape[1]), lower=True)

        if full_covar:
            Sig = np.dot(L_inv_of_Sig_inv.T, L_inv_of_Sig_inv)
            return mu, Sig
        else:
            return mu, Sig_inv, L_inv_of_Sig_inv

    def fit(self, X, y):
        """
        :param X: Positions (N x 2)
        :param y: labels (N,)
        """

        # If first run, set m0, S0i
        if self.scan_no == 0:
            self.mu = np.zeros((self.grid.shape[0] + 1))
            self.Sig_inv = 0.0001 * np.eye((self.grid.shape[0] + 1)) #0.01 for sim, 0
            self.n_iter = 10
        else:
            self.n_iter = 1

        epsilon = 1
        X_orig = copy.copy(X)

        for i in range(self.n_iter):
            X = self.__sparse_features(X_orig)

            # E-step: update Q(w)
            self.mu, self.Sig_inv, self.Ri = self.__calc_posterior(X, y, epsilon, self.mu, self.Sig_inv)

            # M-step: update epsilon
            XMX = np.dot(X, self.mu)**2
            XSX = np.sum(np.dot(X, self.Ri.T) ** 2, axis=1)
            epsilon = np.sqrt(XMX + XSX)

            # Calculate loss, if specified
            if self.calc_loss is True:
                print("  scan={}, iter={} => loss={:.1f}".format(self.scan_no, i,
                      self.__calc_loss(X, np.zeros((self.grid.shape[0] + 1)), 0.01*np.eye((self.grid.shape[0] + 1)),
                        self.mu, self.Sig_inv, self.Ri, epsilon)))

        self.intercept_ = [0]
        coef_, sigma_ = self.__calc_posterior(X, y, epsilon, self.mu, self.Sig_inv, True)

        self.intercept_ = 0
        self.coef_[0] = coef_
        self.sigma_[0] = sigma_
        self.coef_ = np.asarray(self.coef_)
        self.scan_no += 1

    def predict_proba(self, X_q):
        """
        :param X_q: Query positions (N x 2)
        :return: (free prob, occupance prob)
        """

        X_q = self.__sparse_features(X_q)#[:, 1:]
        scores = self.decision_function(X_q)

        sigma = np.asarray([np.sum(np.dot(X_q, s) * X_q, axis=1) for s in self.sigma_])
        ks = 1. / (1. + np.pi * sigma / 8) ** 0.5
        probs = expit(scores.T * ks).T
        if probs.shape[1] == 1:
            probs = np.hstack([1 - probs, probs])
        else:
            probs /= np.reshape(np.sum(probs, axis=1), (probs.shape[0], 1))
        return probs


