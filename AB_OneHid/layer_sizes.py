def layer_sizes(X, Y):
    """

    :param X: input dataset of shape (input size, number of examples)
    :param Y: labels of shape (output size, number of examples)
    :return: sizes of input, hidden and output layers -- n_x, n_h, n_y
    """

    n_x = X[:, 0].size
    n_h = 4
    n_y = Y[:, 0].size

    return (n_x, n_h, n_y)