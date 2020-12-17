# coding=utf-8

"""

@author: six zhang
"""
from connection import Connection


class Perceptron(Connection):

    def __init__(self, input_size, out_size, activator):
        Connection.__init__(self, input_size, out_size, activator)

    def train(self, inputs, labels, rate, epoch):
        """感知机训练: 1.预测；2.计算损失；3.更新权重。
        Args:
            inputs: 训练语料的输入;
            labels: 训练语料输入对应的输出;
            rate: 训练的rate;
            epoch: 训练epoch。
        """
        for i in range(epoch):
            for sample, label in zip(inputs, labels):
                out = self.forward(sample)  # 预测
                delta = out - label  # 损失
                self.backward(sample, delta, rate)  # 更新权重
