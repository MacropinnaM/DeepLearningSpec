import numpy as np


def initialize_parameters_random(layers_dims):
    """
    Initialize parameters randomly
    :param layers_dims: list containing the size of each layer
    :return: dictionary containing your parameters "W1", "b1", ..., "WL", "bL" -- params
    """

    np.random.seed(8)
    params = {}
    L = len(layers_dims)  # integer representing the number of layers

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
        params['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return params