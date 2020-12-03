# coding=utf-8

"""
@author: six zhang
"""

import numpy as np

from bin.networks import Connection
from utils.model import save_2_pickle, load_pickle_model


class FullConns(object):
    """全连接网络"""

    def __init__(self, first_input_size, activator):
        """
        全连接网络初始化
        Args:
            first_input_size: 全连接网络输入层矩阵的size；
            activator: 激活函数，当添加隐藏层没有指定激活函数时，就用初始化的激活函数。
        """
        self.input_size = first_input_size
        self.activator = activator
        self.conns = []

    def _calc_delta(self, label):
        """计算误差"""
        out = self.conns[-1].output
        delta = self.conns[-1].activator.backward(out)*(out-label)  # 输出层误差计算公式
        for conn in self.conns[::-1]:
            conn.delta = delta
            delta = self.conns[-1].activator.backward(conn.input)*np.dot(conn.weight.T, delta)  # 其他层的误差计算公式

    def _update(self, rate):
        """跟新权重矩阵和bias，梯度下降过程"""
        for conn in self.conns:
            conn.weight -= rate*np.dot(conn.delta, conn.input.T)
            conn.bias += rate*conn.delta

    def add_conn(self, size, activator=None):
        """增加隐藏层或者输出层
        Args:
            size: 隐藏层或者输出层的size;
            activator: 当前层的激活函数，如果为None，则用初始化的激活函数
        """
        input_size = self.input_size if not self.conns else self.conns[-1].out_size
        self.conns.append(Connection(input_size, size, (activator or self.activator)))

    def predict(self, x):
        """预测"""
        out = x
        for conn in self.conns:
            conn.process(out)  # 正向计算
            out = conn.output
        return out

    def train(self, inputs, labels, rate, epoch):
        """全连接网络训练: 1.预测；2.计算损失；3.更新权重。
        Args:
            inputs: 训练语料的输入;
            labels: 训练语料输入对应的输出;
            rate: 训练的rate;
            epoch: 训练epoch。
        """
        for i in range(epoch):
            for sample, label in zip(inputs, labels):
                self.predict(sample)
                self._calc_delta(label)
                self._update(rate)

    def save(self, path):
        """将网络保存为模型"""
        save_2_pickle(self, path)

    @staticmethod
    def load(path):
        """加载全连接网络保存的模型"""
        return load_pickle_model(path)
