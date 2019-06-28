import tensorflow as tf


def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and
    the jth column corresponds to the jth training example.
    So if example j had a label i. Then entry (i,j) will be 1.
    :param labels: vector containing the labels
    :param C: number of classes, the depth of the one hot dimension
    :return: one hot matrix
    """

    C = tf.constant(C, name="C")
    one_hot_matrix = tf.one_hot(labels, C, axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()

    return one_hot