# coding=utf-8

"""
@author: six zhang
"""

import numpy as np


class Connection(object):
    """神经网络中的链接抽象，前向传播过程"""

    def __init__(self, input_size, out_size, activator):
        """初始化
        Args:
            input_size: 链接层的输入
            out_size: 链接层输出
            activator: 激活函数
        """
        self.input = None  # 链接层的输入，是一个numpy.array矩阵对象
        self.output = None  # 链接层的输出，也是一个numpy.array对象
        self.delta = None  # 损失，也是一个矩阵
        self.activator = activator  # 激活函数
        self.out_size = out_size
        self.input_size = input_size
        self.bias = np.zeros((out_size, 1))  # 初始化一个bias矩阵
        self.weight = np.random.uniform(-0.1, 0.1, (out_size, self.input_size))  # 初始化一个weight矩阵

    def process(self, sample):
        """前向传播过程
        Args:
            sample: 输入的样本，一个矩阵
        """
        self.input = sample
        shape = sample.shape
        if 3 == len(shape):
            input_size = sample.shape[0]*sample.shape[1]*sample.shape[2]
            self.input = sample.reshape((input_size, 1))
        y = np.dot(self.weight, self.input) + self.bias  # 线性计算
        self.output = self.activator.forward(y)  # 激活函数
