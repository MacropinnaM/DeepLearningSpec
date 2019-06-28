from BD_Opti.forward_propagation import forward_propagation


def predict_dec(parameters, X):
    """
    Used for plotting decision boundary
    :param parameters: dictionary containing the parameters
    :param X: input data of size (m, K)
    :return: vector of predictions of our model (red: 0 / blue: 1) -- preds
    """

    a3, cache = forward_propagation(X, parameters)
    preds = (a3 > 0.5)

    return preds