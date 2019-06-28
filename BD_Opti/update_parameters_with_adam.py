import numpy as np


def update_parameters_with_adam(parameters, grads, v, s, t, lr=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using Adam
    :param parameters: dictionary containing parameters Wl and bl
    :param grads: dictionary containing your gradients for each parameters dWl and dbl
    :param v: Adam variable, moving average of the first gradient, python dictionary
    :param s: Adam variable, moving average of the squared gradient, python dictionary
    :param t: "time"
    :param lr: the learning rate, scalar
    :param beta1: exponential decay hyperparameter for the first moment estimates
    :param beta2: exponential decay hyperparameter for the second moment estimates
    :param epsilon: hyperparameter preventing division by zero in Adam updates
    :return: dictionary with updated parameters -- parameters
             dictionary with Adam variable, moving average of the first gradient -- v
             dictionary with Adam variable, moving average of the squared gradient -- s
    """

    L = len(parameters) // 2  # number of layers in the neural networks
    v_corrected = {}  # Initializing first moment estimate, python dictionary
    s_corrected = {}  # Initializing second moment estimate, python dictionary

    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * grads['dW' + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * grads['db' + str(l + 1)]

        # Compute bias-corrected first moment estimate
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - pow(beta1, t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - pow(beta1, t))

        # Moving average of the squared gradients
        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(grads['dW' + str(l + 1)], 2)
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(grads['db' + str(l + 1)], 2)

        # Compute bias-corrected second raw moment estimate
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - pow(beta2, t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - pow(beta2, t))

        # Update parameters
        parameters["W" + str(l + 1)] =\
            parameters["W" + str(l + 1)] - lr * np.divide(v_corrected["dW" + str(l + 1)],
                                                          np.sqrt(s_corrected["dW" + str(l + 1)]) + epsilon)
        parameters["b" + str(l + 1)] =\
            parameters["b" + str(l + 1)] - lr * np.divide(v_corrected["db" + str(l + 1)],
                                                          np.sqrt(s_corrected["db" + str(l + 1)]) + epsilon)

    return parameters, v, s