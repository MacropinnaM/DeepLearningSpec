import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(model, X, y, h=0.01):
    """
    Plot decision boundary for the model
    :param model:
    :param X: training examples
    :param y: labels
    :param h: distance between points in the grid
    :return: the contour and training examples -- plot
    """

    # Generate the grid
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2'), plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Spectral)
    plt.show()