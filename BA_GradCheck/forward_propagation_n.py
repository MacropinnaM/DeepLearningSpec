import numpy as np

from BA_GradCheck.relu import relu
from BA_GradCheck.sigmoid import sigmoid


def forward_propagation_n(X, Y, parameters):
    """
    Implements the forward propagation (and computes the cost)
    :param X: training set for m examples
    :param Y: labels for m examples
    :param parameters: dictionary containing parameters "W1", "b1", "W2", "b2", "W3", "b3"
    :return: the cost function (logistic cost for one example) -- cost
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

    # Cost
    logprobs = np.multiply(-np.log(A3), Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1. / m * np.sum(logprobs)

    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)

    return cost, cache