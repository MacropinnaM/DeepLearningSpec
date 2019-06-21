import numpy as np


def initialize_parameters(n_x, n_h, n_y):
    """

    :param n_x: size of the input layer
    :param n_h: size of the hidden layer
    :param n_y: size of the output layer
    :return: python dictionary containing your parameters -- params
             W1 -- weight matrix of shape (n_h, n_x)
             b1 -- bias vector of shape (n_h, 1)
             W2 -- weight matrix of shape (n_y, n_h)
             b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(8)

    W1, b1 = np.random.randn(n_h, n_x) * 0.01, np.random.randn(n_h, 1) * 0.01
    W2, b2 = np.random.randn(n_y, n_h) * 0.01, np.random.randn(n_y, 1) * 0.01

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters