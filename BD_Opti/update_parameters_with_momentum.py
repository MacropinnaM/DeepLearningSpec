def update_parameters_with_momentum(parameters, grads, v, beta, lr):
    """
    Update parameters using Momentum
    :param parameters: dictionary containing parameters Wl and bl
    :param grads: dictionary containing gradients for each parameters dWl and dbl
    :param v: dictionary with current velocities for dWl and dbl
    :param beta: the momentum hyperparameter, scalar
    :param lr: the learning rate, scalar
    :return: dictionary containing updated parameters -- parameters
             dictionary containing updated velocities -- v
    """

    L = len(parameters) // 2  # number of layers in the neural networks

    for l in range(L):
        v["dW" + str(l + 1)] = beta * v["dW" + str(l + 1)] + (1 - beta) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta * v["db" + str(l + 1)] + (1 - beta) * grads["db" + str(l + 1)]

        parameters["W" + str(l + 1)] -= lr * v["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] -= lr * v["db" + str(l + 1)]

    return parameters, v