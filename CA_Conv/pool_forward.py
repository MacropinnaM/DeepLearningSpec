import numpy as np


def pool_forward(A_prev, hparameters, mode="max"):
    """
    Implements the forward pass of the pooling layer
    :param A_prev: Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param hparameters: dictionary containing "f" and "stride"
    :param mode: the pooling mode ("max" or "average")
    :return: output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C) -- A
             cache used in the backward pass of the pooling layer, contains the input and hparameters -- cache
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):  # loop over the training examples
        for h in range(n_H):  # loop on the vertical axis of the output volume
            for w in range(n_W):  # loop on the horizontal axis of the output volume
                for c in range(n_C):  # loop over the channels of the output volume
                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    # Use the corners to define the current slice on the ith training example of A_prev, channel c
                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    # Compute the pooling operation on the slice. Use an if statment to differentiate the modes
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    cache = (A_prev, hparameters)
    assert (A.shape == (m, n_H, n_W, n_C))

    return A, cache