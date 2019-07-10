import tensorflow as tf


def create_placeholders(n_H0, n_W0, n_C0, n_y):
    """
    Creates the placeholders for the tensorflow session
    :param n_H0: scalar, height of an input image
    :param n_W0: scalar, width of an input image
    :param n_C0: scalar, number of channels of the input
    :param n_y: scalar, number of classes
    :return: placeholder for the data and labels -- X and Y
    """

    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])

    return X, Y