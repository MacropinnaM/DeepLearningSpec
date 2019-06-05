import numpy as np


def initialize_with_zeros(dim):
    """
    Creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
    :param dim: size of the w vector we want (or number of parameters in this case)
    :return: vector of shape (dim, 1) -- w,
             scalar (corresponds to the bias) -- b
    """
    w = np.zeros((dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b