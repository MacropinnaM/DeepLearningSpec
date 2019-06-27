import matplotlib.pyplot as plt

from BB_Ini.backward_propagation import backward_propagation
from BB_Ini.compute_cost import compute_cost
from BB_Ini.forward_propagation import forward_propagation
from BB_Ini.initialize_parameters_he import initialize_parameters_he
from BB_Ini.initialize_parameters_random import initialize_parameters_random
from BB_Ini.initialize_parameters_xavier import initialize_parameters_xavier
from BB_Ini.initialize_parameters_zeros import initialize_parameters_zeros
from BB_Ini.update_parameters import update_parameters


def model(X, Y, lr=0.01, num_iterations=15000, print_cost=True, ini="zero"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.
    :param X: input data, of shape (2, number of examples)
    :param Y: true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    :param lr: learning rate for gradient descent
    :param num_iterations: number of iterations to run gradient descent
    :param print_cost: if True, print the cost every 1000 iterations
    :param ini: flag to choose which initialization to use ("zeros", "random" or "he")
    :return: parameters learnt by the model -- params
    """

    costs = []  # to keep track of the loss
    layers_dims = [X.shape[0], 10, 5, 1]

    # Initialize parameters dictionary.
    if ini == "zeros":
        params = initialize_parameters_zeros(layers_dims)
    elif ini == "random":
        params = initialize_parameters_random(layers_dims)
    elif ini == "he":
        params = initialize_parameters_he(layers_dims)
    elif ini == "xavier":
        params = initialize_parameters_xavier(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):
        a3, cache = forward_propagation(X, params)
        cost = compute_cost(a3, Y)
        grads = backward_propagation(X, Y, cache)
        params = update_parameters(params, grads, lr)

        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)

    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(lr))
    plt.show()

    return params