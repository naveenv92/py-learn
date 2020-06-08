'''Linear Regression'''

from pylearn import LinearRegression
import numpy as np

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