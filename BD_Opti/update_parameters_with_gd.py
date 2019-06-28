def update_parameters_with_gd(parameters, grads, lr):
    """
    Update parameters using one step of gradient descent
    :param parameters: dictionary containing parameters to be updated
    :param grads: dictionary containing gradients to update each parameters
    :param lr: the learning rate, scalar
    :return: dictionary containing updated parameters -- parameters
    """

    L = len(parameters) // 2  # number of layers in the neural networks

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - lr * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - lr * grads["db" + str(l + 1)]

    return parameters