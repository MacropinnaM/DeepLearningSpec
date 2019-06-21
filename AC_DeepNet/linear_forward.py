import numpy as np


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation
    :param A: activations from previous layer (or input data): (size of previous layer, number of examples)
    :param W: weights matrix: numpy array of shape (size of current layer, size of previous layer)
    :param b: bias vector, numpy array of shape (size of the current layer, 1)
    :return: the input of the activation function, also called pre-activation parameter -- Z
             a python dictionary containing "A", "W" and "b" for backprop -- cache
    """

    Z = np.dot(W, A) + b

    assert (Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache