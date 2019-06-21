import numpy as np

from AA_LogReg.sigmoid import sigmoid


def forward_propagation(X, parameters):
    """
    Implement forward propagation step.
    :param X: input data of size (n_x, m)
    :param parameters: python dictionary containing parameters (output of initialization function)
    :return: the sigmoid output of the second activation -- A2
             a dictionary containing "Z1", "A1", "Z2" and "A2" -- cache
    """

    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]

    # Forward Propagation
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert (A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache