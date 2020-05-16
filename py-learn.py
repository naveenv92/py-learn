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
        X = np.concatenate((np.ones(shape=(X.shape[0], 1)), X), axis=1)
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
        X = np.concatenate((np.ones(shape=(X.shape[0], 1)), X), axis=1)
        return np.dot(X, self.weights)



if __name__ == '__main__':

    # Generate some data with noise
    w0_samp = 4
    w1_samp = 3
    x = np.linspace(-2, 4, 100)
    np.random.seed(10242)
    y = w0_samp + w1_samp*x + 3*np.random.normal(size=x.size)

    # Initialize variables
    X = x.reshape(len(x), 1)
    y = y.reshape(len(x), 1)

    fit = LinearRegression()
    fit.fit(X, y, 0.01, 1000, True)
    print(fit.weights)
    print(fit.predict(np.array([[3], [4], [5]])))
