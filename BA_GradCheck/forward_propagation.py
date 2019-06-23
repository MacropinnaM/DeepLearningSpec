def forward_propagation(x, theta):
    """
    Implement the linear forward propagation (compute J): J(theta) = theta * x
    :param x: a real-valued input
    :param theta: our parameter, a real number as well
    :return: the value of function J -- J
    """

    J = theta * x

    return J