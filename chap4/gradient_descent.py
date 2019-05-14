import numpy as np
from chap3.out_fuction import softmax
from chap4.loss_function import cross_entropy
from chap4.gradient import numerical_gradient_ndim


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)

        loss = cross_entropy(y, t)

        return loss


net = simpleNet()
print(net.W)

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))

t = np.array([0, 0, 1])
loss = net.loss(x, t)
print(loss)


def f(W):
    print("x: {}".format(x))
    print("t: {}".format(t))
    return net.loss(x, t)

f = lambda w: net.loss(x, t)
dw = numerical_gradient_ndim(f, net.W)
print(dw)
