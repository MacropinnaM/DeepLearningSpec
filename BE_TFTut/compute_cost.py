import tensorflow as tf


def compute_cost(Z3, Y):
    """
    Computes the cost
    :param Z3: output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    :param Y: "true" labels vector placeholder, same shape as Z3
    :return: tensor of the cost function -- cost
    """

    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    return cost