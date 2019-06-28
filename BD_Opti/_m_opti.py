import matplotlib.pyplot as plt

from BD_Opti.backward_propagation import backward_propagation
from BD_Opti.compute_cost import compute_cost
from BD_Opti.forward_propagation import forward_propagation
from BD_Opti.initialize_adam import initialize_adam
from BD_Opti.initialize_parameters import initialize_parameters
from BD_Opti.initialize_velocity import initialize_velocity
from BD_Opti.random_mini_batches import random_mini_batches
from BD_Opti.update_parameters_with_adam import update_parameters_with_adam
from BD_Opti.update_parameters_with_gd import update_parameters_with_gd
from BD_Opti.update_parameters_with_momentum import update_parameters_with_momentum


def model(X, Y, layers_dims, optimizer, lr=0.0007, mb_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
    """
    3-layer neural network model which can be run in different optimizer modes
    :param X: input data, of shape (2, number of examples)
    :param Y: true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    :param layers_dims: list, containing the size of each layer
    :param optimizer: optimizer - gd, momentum or adam
    :param lr: the learning rate, scalar
    :param mb_size: the size of a mini batch
    :param beta: momentum hyperparameter
    :param beta1: exponential decay hyperparameter for the past gradients estimates
    :param beta2: exponential decay hyperparameter for the past squared gradients estimates
    :param epsilon: hyperparameter preventing division by zero in Adam updates
    :param num_epochs: number of epochs
    :param print_cost: True to print the cost every 1000 epochs
    :return: dictionary containing your updated parameters -- parameters
    """

    L = len(layers_dims)  # number of layers in the neural networks
    costs = []  # to keep track of the cost
    t = 0  # initializing the counter required for Adam update
    seed = 10  # For grading purposes, so that your "random" minibatches are the same as ours

    # Initialize parameters
    parameters = initialize_parameters(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Optimization loop
    for i in range(num_epochs):

        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mb_size, seed)

        for minibatch in minibatches:
            (minibatch_X, minibatch_Y) = minibatch
            a3, caches = forward_propagation(minibatch_X, parameters)
            cost = compute_cost(a3, minibatch_Y)
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, lr)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, lr)
            elif optimizer == "adam":
                t = t + 1  # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, lr, beta1, beta2, epsilon)

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(lr))
    plt.show()

    return parameters