import numpy as np


def initialize_velocity(parameters):
    """
    Initializes the velocity as a dictionary with keys ("dWL", "dbL") and values (zeros of the same shape)
    :param parameters: dictionary containing the parameters
    :return: dictionary of current velocities: v['dW' + str(l)] for dWl and v['db' + str(l)] for dbl
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}

    # Initialize velocity
    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros((parameters['W' + str(l + 1)].shape[0], parameters['W' + str(l + 1)].shape[1]))
        v["db" + str(l + 1)] = np.zeros((parameters['b' + str(l + 1)].shape[0], parameters['b' + str(l + 1)].shape[1]))

    return v