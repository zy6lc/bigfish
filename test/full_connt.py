# coding=utf-8

"""
@author: six zhang
"""
from bin.networks.full_conn import FullConns
from utils.mnist import translate_label, get_label
from utils.networks import SigmoidActivator


if __name__ == '__main__':
    pass
    # http://yann.lecun.com/exdb/mnist/
    train_images, train_labels, test_images, test_labels = load('your mnist dir, download for url http://yann.lecun.com/exdb/mnist/')

    # train
    network = FullConns(28 * 28, SigmoidActivator())
    network.add_conn(300)
    network.add_conn(10)
    train_labels = [translate_label(x) for x in train_labels]
    network.train(train_images[:600], train_labels[:600], 0.0085, 55)

    for x, y in zip(test_images[:10], test_labels[:10]):
        v = network.predict(x)
        print(y, get_label(v))
