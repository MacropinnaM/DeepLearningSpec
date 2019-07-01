import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

from BE_TFTut.compute_cost import compute_cost
from BE_TFTut.create_placeholders import create_placeholders
from BE_TFTut.forward_propagation import forward_propagation
from BE_TFTut.initialize_parameters import initialize_parameters
from BE_TFTut.random_mini_batches import random_mini_batches


def model(X_train, Y_train, X_test, Y_test, lr=0.0001,
          num_epochs=1500, mb_size=32, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX
    :param X_train: training set, of shape (input size = 12288, number of training examples = 1080)
    :param Y_train: test set, of shape (output size = 6, number of training examples = 1080)
    :param X_test: training set, of shape (input size = 12288, number of training examples = 120)
    :param Y_test: test set, of shape (output size = 6, number of test examples = 120)
    :param lr: learning rate of the optimization
    :param num_epochs: number of epochs of the optimization loop
    :param mb_size: size of a minibatch
    :param print_cost: True to print the cost every 100 epochs
    :return: parameters learnt by the model - params
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost

    X, Y = create_placeholders(n_x, n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / mb_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, mb_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(lr))
        plt.show()

        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters