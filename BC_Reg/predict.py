import numpy as np

from BC_Reg.forward_propagation import forward_propagation


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network
    :param X: data set of examples you would like to label
    :param y: parameters of the trained model
    :param parameters: parameters of the trained model
    :return: predictions for the given dataset X -- preds
    """

    m = X.shape[1]
    preds = np.zeros((1, m), dtype=np.int)

    # Forward propagation
    a3, caches = forward_propagation(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0, i] > 0.5:
            preds[0, i] = 1
        else:
            preds[0, i] = 0

    print("Accuracy: " + str(np.mean((preds[0, :] == y[0, :]))))

    return preds