import tensorflow as tf


def sigmoid(z):
    """
    Computes the sigmoid of z
    :param z: input value, scalar or vector
    :return: the sigmoid of z
    """

    x = tf.placeholder(tf.float32, name="x")
    sigmoid = tf.sigmoid(x)
    sess = tf.Session()
    result = sess.run(sigmoid, feed_dict={x: z})
    sess.close()

    return result