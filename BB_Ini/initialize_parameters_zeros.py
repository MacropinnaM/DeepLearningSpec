import numpy as np


def initialize_parameters_zeros(layers_dims):
    """
    Initialize all parameters to zeros
    :param layers_dims: python array (list) containing the size of each layer
    :return: dictionary containing your parameters "W1", "b1", ..., "WL", "bL" -- params
    """

    params = {}
    L = len(layers_dims)  # number of layers in the network

    for l in range(1, L):
        params['W' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
        params['b' + str(l)] = np.zeros((layers_dims[l], 1))

    return params