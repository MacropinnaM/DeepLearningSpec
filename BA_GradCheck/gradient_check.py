import numpy as np

from BA_GradCheck.backward_propagation import backward_propagation
from BA_GradCheck.forward_propagation import forward_propagation


def gradient_check(x, theta, epsilon=1e-7):
    """
    Implement the backward propagation
    :param x: a real-valued input
    :param theta: parameter, a real number as well
    :param epsilon: tiny shift to the input to compute approximated gradient
    :return: difference between the approximated gradient and the backward propagation gradient -- difference
    """

    # Compute gradapprox using left side of formula (1). epsilon is small enough, you don't need to worry about the limit.
    thetaplus, thetaminus = theta + epsilon, theta - epsilon
    J_plus, J_minus = forward_propagation(x, thetaplus), forward_propagation(x, thetaminus)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)

    # Check if gradapprox is close enough to the output of backward_propagation()
    grad = backward_propagation(x, theta)

    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if difference < 1e-7:
        print("The gradient is correct!")
    else:
        print("The gradient is wrong!")

    return difference