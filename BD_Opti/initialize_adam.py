import numpy as np


def initialize_adam(parameters):
    """
    Initializes v and s as two dictionaries with keys dWl and dbl and values - zeros
    :param parameters: dictionary containing parameters Wl and bl
    :return: dictionary with the exponentially weighted average of the gradient -- v
             dictionary with the exponentially weighted average of the squared gradient -- s
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v = {}
    s = {}

    for l in range(L):
        v["dW" + str(l + 1)] = np.zeros((parameters["W" + str(l + 1)].shape[0], parameters["W" + str(l + 1)].shape[1]))
        v["db" + str(l + 1)] = np.zeros((parameters["b" + str(l + 1)].shape[0], parameters["b" + str(l + 1)].shape[1]))
        s["dW" + str(l + 1)] = np.zeros((parameters["W" + str(l + 1)].shape[0], parameters["W" + str(l + 1)].shape[1]))
        s["db" + str(l + 1)] = np.zeros((parameters["b" + str(l + 1)].shape[0], parameters["b" + str(l + 1)].shape[1]))

    return v, s