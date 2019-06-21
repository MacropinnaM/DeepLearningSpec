def update_parameters(parameters, grads, lr=1.2):
    """
    Updates parameters using the gradient descent update rule given above
    :param parameters: python dictionary containing your parameters
    :param grads: python dictionary containing your gradients
    :param lr: learning rate
    :return: python dictionary containing your updated parameters -- parameters
    """

    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]

    dW1, db1 = grads["dW1"], grads["db1"]
    dW2, db2 = grads["dW2"], grads["db2"]

    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters