import numpy as np

from AC_DeepNet.initialize_parameters_deep import initialize_parameters_deep
from AC_DeepNet.linear_activation_forward import linear_activation_forward
from AC_DeepNet.linear_forward import linear_forward

np.random.seed(8)


# initialize_parameters_deep
parameters = initialize_parameters_deep([5, 4, 3])
cond1 = np.mean(parameters["W1"]).round(4) == .0001
cond2 = np.mean(parameters["b1"]).round(4) == .0
cond3 = np.mean(parameters["W2"]).round(4) == .0001
cond4 = np.mean(parameters["b2"]).round(4) == .0
cond = cond1 & cond2 & cond3 & cond4
if cond:
    print("Test initialize_parameters_deep is OK")
else:
    print("Test initialize_parameters_deep FAILS")


# linear_forward
A = np.array([[-1.02387576, 1.12397796],
              [-1.62328545, 0.64667545],
              [-1.74314104, -0.59664964]])
W = np.array([[0.74505627, 1.97611078, -1.24412333]])
b = np.array([[1]])
Z, linear_cache = linear_forward(A, W, b)
cond = np.allclose(Z, [[-0.8019545, 3.85763489]])
if cond:
    print("Test linear_forward is OK")
else:
    print("Test linear_forward FAILS")


# linear_activation_forward
A_prev = np.random.randn(3, 2)
W = np.random.randn(1, 3)
b = np.random.randn(1, 1)
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation="sigmoid")
cond1 = np.allclose(A, [[0.41367206, 0.78630206]])
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
cond2 = np.allclose(A, [[0., 1.30277749]])
cond = cond1 & cond2
if cond:
    print("Test linear_activation_forward is OK")
else:
    print("Test linear_activation_forward FAILS")


#

"""
print(np.mean(___).round(4))

"""
