import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np


"""Neural Network: Feed Forward with Batch
"""


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

from chap3.activation_function import sigmoid
from chap3.out_fuction import softmax

import pickle
def init_network():
    with open('../dataset/sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    y_batch = predict(network, x[i:i + batch_size])
    p = np.argmax(y_batch, axis=1)

    accuracy_cnt += np.sum(p == t[i:i + batch_size])

    if i % 100 == 0:
        print(i)



print("Accuracy: {:.3f}".format(float(accuracy_cnt) / len(x)))