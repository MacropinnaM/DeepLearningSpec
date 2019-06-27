def update_parameters(parameters, grads, lr):
    """
    Update parameters using gradient descent
    :param parameters: dictionary containing your parameters
    :param grads: dictionary containing your gradients for each parameters
    :param lr: the learning rate, scalar
    :return: dictionary containing your updated parameters -- parameters
    """

    n = len(parameters) // 2  # number of layers in the neural networks

    for k in range(n):
        parameters["W" + str(k + 1)] = parameters["W" + str(k + 1)] - lr * grads["dW" + str(k + 1)]
        parameters["b" + str(k + 1)] = parameters["b" + str(k + 1)] - lr * grads["db" + str(k + 1)]

    return parameters