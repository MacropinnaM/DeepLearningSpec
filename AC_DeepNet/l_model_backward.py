import numpy as np

from AC_DeepNet.linear_activation_backward import linear_activation_backward


def l_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    :param AL: probability vector, output of the forward propagation (L_model_forward())
    :param Y: true "label" vector (containing 0 if non-cat, 1 if cat)
    :param caches: list of caches containing every cache of linear_activation_forward() with "relu" and
                   the cache of linear_activation_forward() with "sigmoid"
    :return: dictionary with the gradients -- grads
    """

    grads = {}
    L = len(caches)  # the number of layers
    Y = Y.reshape(AL.shape)

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    # Lth layer (SIGMOID -> LINEAR) gradients
    # Inputs: "dAL, current_cache"
    # Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                      current_cache,
                                                                                                      "sigmoid")

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients
        # Inputs: "grads["dA" + str(l + 1)], current_cache"
        # Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)]
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)],
                                                                    current_cache,
                                                                    "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads