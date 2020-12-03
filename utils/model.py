# coding=utf-8

"""
@author: six zhang
"""

try:
    import cPickle as pickle
except ImportError:
    import pickle


def save_2_pickle(a_object, path):
    f = open(path, 'wb')
    pickle.dump(a_object, f)
    f.close()
    print('Model saved in %s' % path)


def load_pickle_model(path):
    f = open(path, 'rb')
    a_object = pickle.load(f)
    f.close()
    return a_object
