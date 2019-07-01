import tensorflow as tf


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    :param n_x: scalar, size of an image vector (num_px * num_px * rgb)
    :param n_y: scalar, number of classes
    :return: placeholder for the data input, of shape [n_x, None] and dtype "float" -- X
             placeholder for the input labels, of shape [n_y, None] and dtype "float" -- Y
    """

    X = tf.placeholder(dtype=tf.float32, shape=[n_x, None])
    Y = tf.placeholder(dtype=tf.float32, shape=[n_y, None])

    return X, Y