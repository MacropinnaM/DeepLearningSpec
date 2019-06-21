import numpy as np

from AB_OneHid.backward_propagation import backward_propagation
from AB_OneHid.compute_cost import compute_cost
from AB_OneHid.forward_propagation import forward_propagation
from AB_OneHid.initialize_parameters import initialize_parameters
from AB_OneHid.layer_sizes import layer_sizes
from AB_OneHid.update_parameters import update_parameters


def onehid(X, Y, n_h,
           num_iterations=10000, print_cost=False):
    """
    Implement one-hidden neural network model
    :param X: dataset of shape (2, number of examples)
    :param Y: labels of shape (1, number of examples)
    :param n_h: size of the hidden layer
    :param num_iterations: Number of iterations in gradient descent loop
    :param print_cost: if True, print the cost every 1000 iterations
    :return: parameters learnt by the model -- parameters
    """

    np.random.seed(8)

    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    parameters = initialize_parameters(n_x, n_h, n_y)

    # Gradient descent
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters