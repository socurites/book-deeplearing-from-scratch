"""Step function
"""
import numpy as np


def step_function(x):
    y = x > 0
    return y.astype(np.int)


x = np.array([-1.0, 1.0, 2.0])
y = step_function(x)
print(y)

# Plot
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


"""Sigmoid function
"""
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
y = sigmoid(x)
print(y)

# Plot
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


"""ReLU fuction
"""
def relu(x):
    return np.maximum(0, x)

x = np.array([-1.0, 1.0, 2.0])
y = relu(x)
print(y)

# Plot
x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

import matplotlib.pyplot as plt
plt.plot(x, y)
plt.ylim(-0.1, 5.1)
plt.show()


"""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""
def identity_function(x):
    return x