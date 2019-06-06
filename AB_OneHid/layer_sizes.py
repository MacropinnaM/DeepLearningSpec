def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    n_x = X[:, 0].size  # size of input layer
    n_h = 4
    n_y = Y[:, 0].size  # size of output layer

    return (n_x, n_h, n_y)