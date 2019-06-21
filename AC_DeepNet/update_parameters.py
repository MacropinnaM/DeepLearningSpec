def update_parameters(parameters, grads, lr):
    """
    Update parameters using gradient descent
    :param parameters: python dictionary containing your parameters
    :param grads: python dictionary containing your gradients, output of L_model_backward
    :param lr: learning rate
    :return:
    """

    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - lr * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - lr * grads["db" + str(l + 1)]

    return parameters