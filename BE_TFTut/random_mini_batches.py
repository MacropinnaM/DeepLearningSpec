import numpy as np
import math


def random_mini_batches(X, Y, mb_size=64, seed=0):
    """

    :param X: input data, of shape (input size, number of examples)
    :param Y: true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    :param mb_size: size of the mini-batches, integer
    :param seed: parameter for initialization of the pseudo-random number generator
    :return: list of synchronous (mini_batch_X, mini_batch_Y) -- mini_batches
    """

    m = X.shape[1]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m / mb_size)  # number of mini batches of size mb_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mb_size: k * mb_size + mb_size]
        mini_batch_Y = shuffled_Y[:, k * mb_size: k * mb_size + mb_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mb_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mb_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mb_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches