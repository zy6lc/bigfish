# coding=utf-8

"""
@author: six zhang
"""

import os
import gzip
import numpy as np


# http://yann.lecun.com/exdb/mnist/


def load(data_folder, height=784, width=1):
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        paths.append(os.path.join(data_folder, fname))

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), height, width)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), height, width)

    return x_train, y_train, x_test, y_test


def get_label(vec):
    index = 0
    value = 0
    for i in range(len(vec)):
        if vec[i] > value:
            value = vec[i]
            index = i
    return index


def translate_label(label):
    new = []
    for _ in range(label):
        new.append(0.1)
    new.append(0.9)
    for _ in range(10-len(new)):
        new.append(0.1)
    return np.array(new).reshape((10, 1))


def translate_train(train):
    new = []
    for ont in train:
        for t in ont:
            new.append(t)
    return np.array(new)
