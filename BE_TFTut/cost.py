import tensorflow as tf


def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy
    :param logits: vector containing z, output of the last linear unit (before the final sigmoid activation)
    :param labels: vector of labels y (1 or 0)
    :return: runs the session of the cost
    """

    z, y  = tf.placeholder(tf.float32, name="z"), tf.placeholder(tf.float32, name="y")
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)
    sess = tf.Session()
    cost = sess.run(cost, feed_dict={z: logits, y: labels})
    sess.close()

    return cost