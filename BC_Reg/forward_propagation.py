import numpy as np

from BC_Reg.relu import relu
from BC_Reg.sigmoid import sigmoid


def forward_propagation(X, parameters):
    """
    Implements the forward propagation (and computes the cost)
    :param X: training set for m examples
    :param parameters: dictionary containing parameters "W1", "b1", "W2", "b2", "W3", "b3"
    :return: activation of the last layer -- A3
             all computed parameters -- cache
    """

    m = X.shape[1]

    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache