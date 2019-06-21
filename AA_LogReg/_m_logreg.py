import numpy as np
from AA_LogReg.initialize_with_zeros import initialize_with_zeros
from AA_LogReg.optimize import optimize
from AA_LogReg.predict import predict


def logreg(X_train, Y_train, X_test, Y_test,
          num_iterations=2000, lr=0.5, print_cost=False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    :param X_train: training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    :param Y_train: training labels represented by a numpy array (vector) of shape (1, m_train)
    :param X_test: test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    :param Y_test: test labels represented by a numpy array (vector) of shape (1, m_test)
    :param num_iterations: the number of iterations to optimize the parameters
    :param lr: the learning rate used in the update rule of optimize()
    :param print_cost: "true" to print the cost every 100 iterations
    :return: dictionary containing information about the model -- d
    """

    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, lr, print_cost)
    w, b = parameters["w"], parameters["b"]

    Y_pred_train = predict(w, b, X_train)
    Y_pred_test = predict(w, b, X_test)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_train": Y_pred_train,
         "Y_prediction_test": Y_pred_test,
         "w": w,
         "b": b,
         "learning_rate": lr,
         "num_iterations": num_iterations}

    return d