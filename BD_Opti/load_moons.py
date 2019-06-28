import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

np.random.seed(3)

def load_moons(n_samples, noise):
    """
    Create dataset with sklearn's make_moons
    :param n_samples: number of examples
    :param noise: noise
    :return: dataset of examples with labels -- train_X and train_Y
    """
    """
     
    :return: 
    """

    train_X, train_Y = sklearn.datasets.make_moons(n_samples=n_samples, noise=noise)

    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y.ravel(), s=40, cmap=plt.cm.Spectral);
    plt.show()

    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))

    return train_X, train_Y