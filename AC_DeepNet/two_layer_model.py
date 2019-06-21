import numpy as np
import matplotlib.pyplot as plt

from AB_OneHid.initialize_parameters import initialize_parameters
from AC_DeepNet.compute_cost import compute_cost
from AC_DeepNet.linear_activation_backward import linear_activation_backward
from AC_DeepNet.linear_activation_forward import linear_activation_forward
from AC_DeepNet.update_parameters import update_parameters


def two_layer_model(X, Y, layers_dims, lr=0.0075, num_iterations=3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID
    :param X: input data, of shape (n_x, number of examples)
    :param Y: true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    :param layers_dims: dimensions of the layers (n_x, n_h, n_y)
    :param lr: learning rate of the gradient descent update rule
    :param num_iterations: number of iterations of the optimization loop
    :param print_cost: If set to True, this will print the cost every 100 iterations
    :return: a dictionary containing W1, W2, b1, and b2 -- parameters
    """

    np.random.seed(8)
    grads = {}
    costs = []  # to keep track of the cost
    (n_x, n_h, n_y) = layers_dims

    parameters = initialize_parameters(n_x, n_h, n_y)
    W1, b1  = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]

    # Gradient descent
    for i in range(0, num_iterations):
        A1, cache1 = linear_activation_forward(X, W1, b1, "relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, "sigmoid")
        cost = compute_cost(A2, Y)
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, "relu")

        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2

        parameters = update_parameters(parameters, grads, lr)
        W1, b1 = parameters["W1"], parameters["b1"]
        W2, b2 = parameters["W2"], parameters["b2"]

        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(lr))
    plt.show()

    return parameters