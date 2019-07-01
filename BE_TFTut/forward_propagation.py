import tensorflow as tf


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    :param X: input dataset placeholder, of shape (input size, number of examples)
    :param parameters: dictionary containing parameters "W1", "b1", "W2", "b2", "W3", "b3"
    :return: the output of the last LINEAR unit -- Z3
    """

    W1, b1 = parameters['W1'], parameters['b1']
    W2, b2 = parameters['W2'], parameters['b2']
    W3, b3 = parameters['W3'], parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3, Z2), b3)

    return Z3