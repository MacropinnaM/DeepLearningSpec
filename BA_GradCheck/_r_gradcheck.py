import numpy as np

from BA_GradCheck.backward_propagation_n import backward_propagation_n
from BA_GradCheck.forward_propagation_n import forward_propagation_n
from BA_GradCheck.gradient_check_n import gradient_check_n

np.random.seed(8)

X = np.random.randn(4,3)
Y = np.array([1, 1, 0])
W1 = np.random.randn(5,4)
b1 = np.random.randn(5,1)
W2 = np.random.randn(3,5)
b2 = np.random.randn(3,1)
W3 = np.random.randn(1,3)
b3 = np.random.randn(1,1)
parameters = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}

cost, cache = forward_propagation_n(X, Y, parameters)
gradients = backward_propagation_n(X, Y, cache)
difference = gradient_check_n(parameters, gradients, X, Y)