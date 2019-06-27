import numpy as np


def initialize_parameters_xavier(layers_dims):
    """
    Initialize parameters with Xavier's rule
    :param layers_dims: list containing the size of each layer
    :return: dictionary containing your parameters "W1", "b1", ..., "WL", "bL" -- params
    """

    np.random.seed(8)
    params = {}
    L = len(layers_dims) - 1  # integer representing the number of layers

    for l in range(1, L + 1):
        params['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(1 / layers_dims[l - 1])
        params['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return params