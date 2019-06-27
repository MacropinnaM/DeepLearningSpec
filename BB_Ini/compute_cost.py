import numpy as np


def compute_cost(a3, Y):
    """
    Implement the cost function
    :param a3: post-activation, output of forward propagation
    :param Y: "true" labels vector, same shape as a3
    :return: value of the cost function -- cost
    """

    m = Y.shape[1]

    logprobs = np.multiply(-np.log(a3), Y) + np.multiply(-np.log(1 - a3), 1 - Y)
    cost = 1. / m * np.nansum(logprobs)

    return cost