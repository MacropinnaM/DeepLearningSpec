import numpy as np
import h5py


def load_cats(data_folder, train_file, test_file):
    """
    Generate features and labels of train and test data and gets the list of classes
    :param data_folder: name of the folder with data files
    :param train_file: name of file with train data
    :param test_file: name of files with test_data
    :return: train set features -- train_x
             train set labels -- train_y
             test set features -- test_x
             test set labels -- test_y
             the list of classes
    """

    train_dataset = h5py.File(data_folder + train_file, "r")
    train_x = np.array(train_dataset["train_set_x"][:])
    train_y = np.array(train_dataset["train_set_y"][:])
    train_y = train_y.reshape((1, train_y.shape[0]))

    test_dataset = h5py.File(data_folder + test_file, "r")
    test_x = np.array(test_dataset["test_set_x"][:])
    test_y = np.array(test_dataset["test_set_y"][:])
    test_y = test_y.reshape((1, test_y.shape[0]))

    classes = np.array(test_dataset["list_classes"][:])

    return train_x, train_y, test_x, test_y, classes