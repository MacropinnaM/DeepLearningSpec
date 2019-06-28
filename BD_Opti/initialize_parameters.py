import numpy as np


def initialize_parameters(layer_dims):
    """

    :param layer_dims: python array (list) containing the dimensions of each layer in our network
    :return: python dictionary containing your parameters "W1", "b1", ..., "WL", "bL" -- parameters
    """

    np.random.seed(8)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters