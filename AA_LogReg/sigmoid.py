import numpy as np


def sigmoid(z):
    """
    Compute the sigmoid of z
    :param z: A scalar or numpy array of any size.
    :return: valut of sigmoid(z) -- s
    """

    s = 1 / (1 + np.exp(-z))

    return s