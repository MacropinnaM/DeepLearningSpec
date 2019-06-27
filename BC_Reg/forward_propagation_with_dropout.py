import numpy as np

from BC_Reg.relu import relu
from BC_Reg.sigmoid import sigmoid


def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    :param X: input dataset, of shape (2, number of examples)
    :param parameters: dictionary containing the parameters "W1", "b1", "W2", "b2", "W3", "b3"
    :param keep_prob: probability of keeping a neuron active during drop-out, scalar
    :return: last activation value, output of the forward propagation, of shape (1,1)  -- A3
             tuple, information stored for computing the backward propagation -- cache
    """

    np.random.seed(8)

    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    W3, b3 = parameters["W3"], parameters["b3"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = D1 < keep_prob  # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A1 = A1 * D1  # Step 3: shut down some neurons of A1
    A1 = A1 / keep_prob  # Step 4: scale the value of neurons that haven't been shut down

    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])  # Step 1: initialize matrix D1 = np.random.rand(..., ...)
    D2 = D2 < keep_prob  # Step 2: convert entries of D1 to 0 or 1 (using keep_prob as the threshold)
    A2 = A2 * D2  # Step 3: shut down some neurons of A1
    A2 = A2 / keep_prob  # Step 4: scale the value of neurons that haven't been shut down

    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3, cache