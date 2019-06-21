from AB_OneHid.forward_propagation import forward_propagation


def predict(parameters, X):
    """
    Predict a class for each example in X, using the learned parameters
    :param parameters: python dictionary containing your parameters
    :param X: input data of size (n_x, m)
    :return: vector of predictions of our model (red: 0 / blue: 1) -- preds
    """

    A2, cache = forward_propagation(X, parameters)
    preds = (A2 > 0.5)

    return preds