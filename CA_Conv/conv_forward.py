import numpy as np

from CA_Conv.conv_single_step import conv_single_step
from CA_Conv.zero_pad import zero_pad


def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    :param A_prev: output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param W: Weights, numpy array of shape (f, f, n_C_prev, n_C)
    :param b: Biases, numpy array of shape (1, 1, 1, n_C)
    :param hparameters: python dictionary containing "stride" and "pad"
    :return: conv output, numpy array of shape (m, n_H, n_W, n_C) -- Z
             cache of values needed for the conv_backward() function -- cache
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters['stride']
    pad = hparameters['pad']

    # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):  # loop over the batch of training examples
        a_prev_pad = A_prev_pad[i]  # Select ith training example's padded activation
        for h in range(n_H):  # loop over vertical axis of the output volume
            for w in range(n_W):  # loop over horizontal axis of the output volume
                for c in range(n_C):  # loop over channels (= #filters) of the output volume
                    # Find the corners of the current "slice"
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell)
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[..., c], b[..., c])

    assert (Z.shape == (m, n_H, n_W, n_C))
    cache = (A_prev, W, b, hparameters)

    return Z, cache