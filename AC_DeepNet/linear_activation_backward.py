from AC_DeepNet.linear_backward import linear_backward
from AC_DeepNet.relu_backward import relu_backward
from AC_DeepNet.sigmoid_backward import sigmoid_backward


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer
    :param dA: post-activation gradient for current layer l
    :param cache: tuple of values (linear_cache, activation_cache) we stored
    :param activation: the activation to be used in this layer: "sigmoid" or "relu"
    :return: gradient of the cost wrt the activation (of the previous layer l-1), same shape as A_prev -- dA_prev
             gradient of the cost wrt W (current layer l), same shape as W -- dW
             gradient of the cost wrt b (current layer l), same shape as b -- db
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db