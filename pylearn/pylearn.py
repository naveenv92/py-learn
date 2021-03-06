"""
py-learn: module of basic ML algorithms
"""

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self):
        self.weights = None
    
    def _cost_MSE(self, X, y, w):
        """
        Calculates mean-squared error of training set
        """
        m = len(y)
        return (1/(2*m))*sum((np.dot(X, w) - y)**2)[0]

    def fit(self, X, y, alpha, num_iters, show_cost=False):
        """
        Uses gradient descent to minimize the cost function
        
        Parameters
            X (np.array): matrix of x values (m x (n+1))
            y (np.array): matrix of y values (m x 1)
            w (np.array): matrix of weight values ((n+1) x 1)
            alpha (float): learning rate
            num_iters (int): number of iterations
        
        Returns
            J (np.array): cost function at each iteration
            w_new (np.array): optimized weights
            w_hist (np.array): weights at each timestep
        """
        
        m = len(y)
        X = np.hstack((np.ones(shape=(X.shape[0], 1)), X))
        w_train = np.zeros(shape=(X.shape[1], 1))
        J = np.zeros(num_iters)
        
        for i in range(num_iters):
            w_train = w_train - (alpha/m)*(np.dot((np.dot(X, w_train) - y).T, X)).T
            J[i] = self._cost_MSE(X, y, w_train)

        if show_cost:
            self._plot_cost(num_iters, J)

        self.weights = w_train

    def _plot_cost(self, num_iters, J):
        fig, ax = plt.subplots()
        ax.plot(range(num_iters), J, linewidth=2)
        ax.set(xlabel='Iterations', ylabel='Cost')
        plt.show()

    def predict(self, X):
        """
        Make predictions
        """
        assert self.weights is not None, 'Model has not been fit'
        X = np.hstack((np.ones(shape=(X.shape[0], 1)), X))
        return np.dot(X, self.weights)

class LogisticRegression:

    def __init__(self):
        self.weights = None

    def _sigmoid(self, z):
        """Returns value of the sigmoid function."""
        return 1/(1 + np.exp(-z))

    def _cost_logistic(self, X, y, w):
        """
        Returns cost function and gradient

        Parameters
            X: m x (n+1) matrix of features
            y: m x 1 vector of labels
            w: (n+1) x 1 vector of weights
        Returns
            cost: value of cost function
            grad: (n+1) x 1 vector of weight gradients
        """

        m = len(y)
        h = self._sigmoid(np.dot(X, w))
        cost = (1/m)*(-np.dot(y.T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h)))
        grad = (1/m)*np.dot(X.T, h - y)
        return cost, grad

    def fit(self, X, y, alpha, num_iters, show_cost=False):
        """
        Uses gradient descent to minimize cost function.
        
        Parameters
            X: m x (n+1) matrix of features
            y: m x 1 vector of labels
            w: (n+1) x 1 vector of weights
            alpha (float): learning rate
            num_iters (int): number of iterations
        Returns
            J: 1 x num_iters vector of costs
            w_new: (n+1) x 1 vector of optimized weights
            w_hist: (n+1) x num_iters matrix of weights
        """

        X = np.hstack((np.ones(shape=(X.shape[0], 1)), X))
        w_train = np.zeros(shape=(X.shape[1], 1))
        J = np.zeros(num_iters)

        for i in range(num_iters):
            cost, grad = self._cost_logistic(X, y, w_train)
            w_train = w_train - alpha*grad
            J[i] = cost

        if show_cost:
            self._plot_cost(num_iters, J)

        self.weights = w_train

    def _plot_cost(self, num_iters, J):
        fig, ax = plt.subplots()
        ax.plot(range(num_iters), J, linewidth=2)
        ax.set(xlabel='Iterations', ylabel='Cost')
        plt.show()