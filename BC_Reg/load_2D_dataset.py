import matplotlib.pyplot as plt
import scipy.io


def load_2D_dataset(comp, file_name):
    """
    Load 2D dataset from the file
    :param file_name: name of file with data
    :return: train and test data and its labels
    """
    data = scipy.io.loadmat('/Users/' + comp + '/PyCharmProjects/DeepLearningSpec/_Data/' + file_name)
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    plt.scatter(train_X[0, :], train_X[1, :], c=train_Y.ravel(), s=40, cmap=plt.cm.Spectral);
    plt.show()

    return train_X, train_Y, test_X, test_Y