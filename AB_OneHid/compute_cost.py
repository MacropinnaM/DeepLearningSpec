import numpy as np


def compute_cost(A2, Y):
    """
    Computes the cross-entropy cost
    :param A2: The sigmoid output of the second activation, of shape (1, number of examples)
    :param Y: "true" labels vector of shape (1, number of examples)
    :return: cross-entropy cost -- cost
    """

    m = Y.shape[1]  # number of example

    logprobs = np.multiply(np.log(A2), Y) + np.multiply(1 - Y, np.log(1 - A2))
    cost = -1 / m * np.sum(logprobs)

    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
    assert (isinstance(cost, float))

    return cost