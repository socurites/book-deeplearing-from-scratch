import numpy as np

X = np.array([1,2])

W = np.array([[1, 3, 5],
              [2, 4, 6]])


print(X.shape)
print(X.T.shape)

Y = np.dot(X, W)
print(Y)
print(Y.shape)


Y = np.dot(X.T, W)
print(Y)
print(Y.shape)