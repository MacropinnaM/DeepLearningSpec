import numpy as np


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    :param Z: array of any shape
    :return: output of sigmoid(z), same shape as Z -- A
    """

    A = 1 / (1 + np.exp(-Z))

    return A