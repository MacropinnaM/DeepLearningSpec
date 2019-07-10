import numpy as np


def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice of the output activation of the previous layer
    :param a_slice_prev: slice of input data of shape (f, f, n_C_prev)
    :param W: Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    :param b: Bias parameters contained in a window - matrix of shape (1, 1, 1)
    :return: a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data -- Z
    """

    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z = float(Z + b)

    return Z