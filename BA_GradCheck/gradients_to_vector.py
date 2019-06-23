import numpy as np


def gradients_to_vector(gradients):
    """
    Roll all gradients dictionary into a single vector satisfying our specific required shape.
    :param gradients: dictionary containing parameters "dW1", "db1", "dW2", "db2", "dW3", "db3"
    :return: vector of parameters -- theta
    """

    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1, 1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta