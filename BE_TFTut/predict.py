import tensorflow as tf

from BE_TFTut.forward_propagation import forward_propagation


def predict(X, parameters):
    """
    This function is used to predict the results of a n-layer neural network
    :param X: data set of examples you would like to label
    :param parameters: parameters of the trained model
    :return: predictions for the given dataset X -- preds
    """

    W1, b1 = tf.convert_to_tensor(parameters["W1"]), tf.convert_to_tensor(parameters["b1"])
    W2, b2 = tf.convert_to_tensor(parameters["W2"]), tf.convert_to_tensor(parameters["b2"])
    W3, b3 = tf.convert_to_tensor(parameters["W3"]), tf.convert_to_tensor(parameters["b3"])

    params = {"W1": W1, "b1": b1,
              "W2": W2, "b2": b2,
              "W3": W3, "b3": b3}

    x = tf.placeholder("float", [12288, 1])

    z3 = forward_propagation(x, params)
    p = tf.argmax(z3)

    with tf.Session() as sess:
        preds = sess.run(p, feed_dict={x: X})

    return preds