# coding=utf-8

"""
@author: six zhang
"""

import numpy as np


class Connection(object):
    """神经网络中的链接抽象"""

    def __init__(self, input_size, out_size, activator):
        """初始化
        Args:
            input_size: 链接层的输入size
            out_size: 链接层输出size
            activator: 激活函数size
        """
        self.delta = None  # 损失，也是一个矩阵
        self.activator = activator  # 激活函数
        self.bias = np.zeros((out_size, 1))  # 初始化一个bias矩阵
        self.weight = np.random.uniform(-1, 1, (out_size, input_size))  # 初始化一个weight矩阵

    def forward(self, sample):
        """前向传播过程
        Args:
            sample: 输入的样本，一个矩阵
        """
        y = np.dot(self.weight, sample) + self.bias  # 线性计算
        return self.activator.forward(y)  # 激活函数

    def backward(self, sample, delta, rate):
        """反向传播过程：传递delta
        Args:
            sample: 统forward一样
            delta: 经过梯度计算之后的损失值
            rate: 梯度速率
        """
        self.delta = delta
        self.weight -= rate * np.dot(self.delta, sample.T)
        self.bias += rate * self.delta
