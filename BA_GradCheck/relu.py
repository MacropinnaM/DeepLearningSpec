import numpy as np


def relu(Z):
    """
    Implement the RELU function
    :param Z: Output of the linear layer, of any shape
    :return: post-activation parameter, of the same shape as Z -- A
    """

    A = np.maximum(0, Z)
    assert (A.shape == Z.shape)

    return A
