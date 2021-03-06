import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
SIZE_IMG = 28

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)


# Plotting
import numpy as np
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


img = x_train[0]
label = t_train[0]

print(label)
print(img.shape)
img = img.reshape(SIZE_IMG, SIZE_IMG)
print(img.shape)

img_show(img)


"""Neural Network: Feed Forward
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

accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x)
    p = np.argmax(y)

    if p == t[i]:
        accuracy_cnt += 1

    if i % 100 == 0:
        print(i)



print("Accuracy: {:.3f}".format(float(accuracy_cnt) / len(x)))