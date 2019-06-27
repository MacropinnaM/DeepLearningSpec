import numpy as np

from BC_Reg.compute_cost import compute_cost


def compute_cost_with_regularization(A3, Y, parameters, lambd):
    """
    Implement the cost function with L2 regularization
    :param A3: post-activation, output of forward propagation, of shape (output size, number of examples)
    :param Y: "true" labels vector, of shape (output size, number of examples)
    :param parameters: dictionary containing parameters of the model
    :param lambd: regularization hyperparameter, scalar
    :return: value of the regularized loss function -- cost
    """

    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = compute_cost(A3, Y)

    L2_reg_cost = 1 / m * lambd / 2 * np.sum(np.square(W1)) + \
                  1 / m * lambd / 2 * np.sum(np.square(W2)) + \
                  1 / m * lambd / 2 * np.sum(np.square(W3))

    cost = cross_entropy_cost + L2_reg_cost

    return cost