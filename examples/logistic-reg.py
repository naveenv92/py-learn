'''Logistic Regression'''

from pylearn import LogisticRegression
import numpy as np

x1 = 5*np.random.random(size=100)
x2 = 5*np.random.random(size=100)

y = np.zeros(len(x1))
boundary = lambda x1, x2: 6 - x1 - 2*x2

# Set y = 1 above decision boundary
for i in range(len(y)):
    if boundary(x1[i], x2[i]) <= 0:
        y[i] = 1
    else:
        y[i] = 0

# Initialize variables
X = np.hstack((x1.reshape(len(x1), 1), x2.reshape(len(x2), 1)))
y = y.reshape(len(y), 1)

fit = LogisticRegression()
fit.fit(X, y, 0.5, 2000, True)
print(fit.weights)