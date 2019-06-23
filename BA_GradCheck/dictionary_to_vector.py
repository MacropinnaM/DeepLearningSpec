import numpy as np


def dictionary_to_vector(parameters):
    """
    Roll all parameters dictionary into a single vector satisfying our specific required shape.
    :param parameters: dictionary containing parameters "W1", "b1", "W2", "b2", "W3", "b3"
    :return: single vector of parameters -- theta
             with keys -- keys
    """

    keys = []
    count = 0

    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:

        new_vector = np.reshape(parameters[key], (-1, 1))
        keys = keys + [key] * new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys