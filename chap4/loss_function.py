import numpy as np

"""MSE
"""
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)


"""XE
"""
def cross_entropy_naive(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def cross_entropy(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    return -np.sum(t * np.log(y)) / batch_size