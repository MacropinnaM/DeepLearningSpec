import numpy as np
import matplotlib.pyplot as plt

from AC_DeepNet.compute_cost import compute_cost
from AC_DeepNet.initialize_parameters_deep import initialize_parameters_deep
from AC_DeepNet.l_model_backward import l_model_backward
from AC_DeepNet.l_model_forward import l_model_forward
from AC_DeepNet.update_parameters import update_parameters


def l_layer_model(X, Y, layers_dims, lr=0.0075,
                  num_iterations=3000, print_cost=False):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID
    :param X: data, numpy array of shape (number of examples, num_px * num_px * 3)
    :param Y: true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    :param layers_dims: list containing the input size and each layer size, of length (number of layers + 1)
    :param lr: learning rate of the gradient descent
    :param num_iterations: number of iterations of the optimization loop
    :param print_cost: if True, it prints the cost every 100 steps
    :return: parameters learnt by the model -- parameters
    """

    np.random.seed(8)
    costs = []  # keep track of cost

    parameters = initialize_parameters_deep(layers_dims)

    # Gradient descent
    for i in range(0, num_iterations):
        AL, caches = l_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = l_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, lr)
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # Plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(lr))
    plt.show()

    return parameters