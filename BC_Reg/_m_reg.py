import matplotlib.pyplot as plt

from BC_Reg.backward_propagation import backward_propagation
from BC_Reg.backward_propagation_with_dropout import backward_propagation_with_dropout
from BC_Reg.backward_propagation_with_regularization import backward_propagation_with_regularization
from BC_Reg.compute_cost import compute_cost
from BC_Reg.compute_cost_with_regularization import compute_cost_with_regularization
from BC_Reg.forward_propagation import forward_propagation
from BC_Reg.forward_propagation_with_dropout import forward_propagation_with_dropout
from BC_Reg.initialize_parameters import initialize_parameters
from BC_Reg.update_parameters import update_parameters


def model(X, Y, lr=0.3, num_iterations=30000, print_cost=True, lambd=0, keep_prob=1):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID
    :param X: input data, of shape (input size, number of examples)
    :param Y: true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    :param lr: learning rate of the optimization
    :param num_iterations: number of iterations of the optimization loop
    :param print_cost: If True, print the cost every 10000 iterations
    :param lambd: regularization hyperparameter, scalar
    :param keep_prob: probability of keeping a neuron active during drop-out, scalar
    :return: parameters learned by the model -- parameters
    """

    grads = {}
    costs = []  # to keep track of the cost
    m = X.shape[1]  # number of examples
    layers_dims = [X.shape[0], 20, 3, 1]

    # Initialize parameters dictionary.
    parameters = initialize_parameters(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob == 1:
            a3, cache = forward_propagation(X, parameters)
        elif keep_prob < 1:
            a3, cache = forward_propagation_with_dropout(X, parameters, keep_prob)

        # Cost function
        if lambd == 0:
            cost = compute_cost(a3, Y)
        else:
            cost = compute_cost_with_regularization(a3, Y, parameters, lambd)

        # Backward propagation.
        assert (lambd == 0 or keep_prob == 1)  # it is possible to use both L2 regularization and dropout,
        # but this assignment will only explore one at a time
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X, Y, cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X, Y, cache, lambd)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(X, Y, cache, keep_prob)

        # Update parameters.
        parameters = update_parameters(parameters, grads, lr)

        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(lr))
    plt.show()

    return parameters