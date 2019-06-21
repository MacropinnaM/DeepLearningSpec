import numpy as np


def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit
    :param dA: post-activation gradient, of any shape
    :param cache: Z' where we stored
    :return: gradient of the cost wrt Z
    """

    Z = cache

    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert (dZ.shape == Z.shape)

    return dZ