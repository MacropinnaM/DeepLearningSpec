import numpy as np


def convert_to_one_hot(Y, C):
    """
    Converts dataset to one hot matrix
    :param Y: dataset converted
    :param C: number of classes
    :return: one hot matrix -- Y
    """
    Y = np.eye(C)[Y.reshape(-1)].T

    return Y