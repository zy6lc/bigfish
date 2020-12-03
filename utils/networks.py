# coding=utf-8

"""
@author: six zhang
"""

import numpy as np


class SigmoidActivator(object):
    """sigmoid激活函数"""

    @staticmethod
    def forward(weighted_input):
        """正向激活计算"""
        return 1.0 / (1.0 + np.exp(-weighted_input))

    @staticmethod
    def backward(output):
        """sigmoid求导"""
        return output * (1 - output)


class ReLuActivator(object):

    @staticmethod
    def forward(weighted_input):
        return weighted_input*(weighted_input>0)

    @staticmethod
    def backward(output):
        return 1.0*(output>0)
