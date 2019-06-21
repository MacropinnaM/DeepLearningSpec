import numpy as np


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit
    :param dA: post-activation gradient, of any shape
    :param cache: Z stored in RELU for backprop
    :return: gradient of the cost with respect to Z -- dZ
    """

    Z = cache
    dZ = np.array(dA, copy=True)

    # When z <= 0, you should set dz to 0 as well.
    dZ[Z <= 0] = 0
    assert (dZ.shape == Z.shape)

    return dZ