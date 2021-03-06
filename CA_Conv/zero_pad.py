import numpy as np


def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image.
    :param X: array of shape (m, n_H, n_W, n_C) representing a batch of m images
    :param pad: integer, amount of padding around each image on vertical and horizontal dimensions
    :return: padded image of shape (m, n_H + 2*pad, n_W + 2*pad, n_C) -- X_pad
    """

    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')

    return X_pad