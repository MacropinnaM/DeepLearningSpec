import numpy as np


def load_flower(n_samples, n_classes, dim, max_ray):
    """
    Generate a "flower" dataset
    :param n_samples: number of samples
    :param n_classes: number of classes
    :param dim: dimensionality
    :param max_ray: maximum ray of the flower
    :return: generated samples -- X
             labels for samples -- Y
    """
    np.random.seed(8)

    N = int(n_samples / n_classes)
    X = np.zeros((n_samples, dim))
    Y = np.zeros((n_samples, 1), dtype='uint8')

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        theta = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        radius = max_ray * np.sin(4 * theta) + np.random.randn(N) * 0.2
        X[ix] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
        Y[ix] = j

    X, Y = X.T, Y.T

    return X, Y