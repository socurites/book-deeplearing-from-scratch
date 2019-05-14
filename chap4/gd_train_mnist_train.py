import numpy as np
from dataset.mnist import load_mnist
from chap4.gd_train_minist_network import TwoLayerNet


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []

# Hyper-parameters
iters_num = 10000
train_size = x_train.shape[0]