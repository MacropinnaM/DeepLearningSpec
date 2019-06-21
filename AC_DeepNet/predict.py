import numpy as np

from AC_DeepNet.l_model_forward import l_model_forward


def predict(X, y, parameters):
    """
    Predict the results of a  L-layer neural network
    :param X: data set of examples you would like to label
    :param y: data set of labels
    :param parameters: parameters of the trained model
    :return: predictions for the given dataset X -- preds
    """

    m = X.shape[1]
    preds = np.zeros((1, m))

    # Forward propagation
    probas, caches = l_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            preds[0, i] = 1
        else:
            preds[0, i] = 0

    print("Accuracy: " + str(np.sum((preds == y) / m)))

    return preds