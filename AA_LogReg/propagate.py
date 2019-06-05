import numpy as np

from AA_LogReg.sigmoid import sigmoid


def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation.
    :param w: weights, a numpy array of size (num_px * num_px * 3, 1)
    :param b: bias, a scalar
    :param X: data of size (num_px * num_px * 3, number of examples)
    :param Y: true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    :return: negative log-likelihood cost for logistic regression -- cost,
             gradient of the loss wrt w, thus same shape as w -- dw,
             gradient of the loss wrt b, thus same shape as b -- db
    """

    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)  # compute activation
    cost = -1 / m * np.sum(np.multiply(np.log(A), Y) + np.multiply(1 - Y, np.log(1 - A)))  # compute cost

    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)

    cost = np.squeeze(cost)

    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost