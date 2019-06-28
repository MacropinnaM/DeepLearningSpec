import tensorflow as tf


def ones(shape):
    """
    Creates an array of ones of dimension shape
    :param shape: shape of the array you want to create
    :return: array containing only ones -- ones
    """

    ones = tf.ones(shape)
    sess = tf.Session()
    ones = sess.run(ones)
    sess.close()

    return ones