import numpy as np
import sklearn.datasets
import matplotlib.pyplot as plt

def load_circles(n_samles_1, n_samples_2, noise):
    """
    Create two circles
    :return: datasets with generated samples and labels
    """
    np.random.seed(2)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=n_samles_1, noise=noise)

    np.random.seed(6)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=n_samples_2, noise=noise)

    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y.ravel(), s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    plt.show()

    return train_X, train_Y, test_X, test_Y