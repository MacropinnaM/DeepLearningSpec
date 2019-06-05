from AA_LogReg.propagate import propagate


def optimize(w, b, X, Y, num_iterations, lr, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    :param w: weights, a numpy array of size (num_px * num_px * 3, 1)
    :param b: bias, a scalar
    :param X: data of shape (num_px * num_px * 3, number of examples)
    :param Y: true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    :param num_iterations: number of iterations of the optimization loop
    :param lr: learning rate of the gradient descent update rule
    :param print_cost: True to print the loss every 100 steps
    :return: dictionary containing the weights w and bias b -- params
             dictionary containing the gradients of the weights and bias wrt the cost function -- grads
             list of all the costs computed during the optimization to plot the learning curve -- costs
    """

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w = w - lr * dw
        b = b - lr * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs