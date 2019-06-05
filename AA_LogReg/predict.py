import numpy as np

from AA_LogReg.sigmoid import sigmoid


def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    :param w: weights, a numpy array of size (num_px * num_px * 3, 1)
    :param b: bias, a scalar
    :param X: data of size (num_px * num_px * 3, number of examples)
    :return: a numpy array (vector) containing all predictions (0/1) for the examples in X -- preds
    """

    m = X.shape[1]
    preds = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] > 0.5:
            preds[0][i] = 1
        else:
            preds[0][i] = 0

    assert (preds.shape == (1, m))

    return preds