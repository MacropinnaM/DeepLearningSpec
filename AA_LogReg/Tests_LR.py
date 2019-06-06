import numpy as np

from AA_LogReg.initialize_with_zeros import initialize_with_zeros
from AA_LogReg.optimize import optimize
from AA_LogReg.predict import predict
from AA_LogReg.propagate import propagate
from AA_LogReg.sigmoid import sigmoid


# sigmoid
z = [0, 2]
s = sigmoid(np.array(z))
cond = np.allclose(s, np.array([0.5, 0.88079708]))
if cond:
    print("Test sigmoid is OK")
else:
    print("Test sigmoid FAILS")


# initialize_with_zeros
dim = 2
w, b = initialize_with_zeros(dim)
cond1 = np.allclose(w, np.array([[0.],
                                 [0.]]))
cond2 = b == 0
cond = cond1 & cond2
if cond:
    print("Test initialize_with_zeros is OK")
else:
    print("Test initialize_with_zeros FAILS")


# propagate
w, b, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
grads, cost = propagate(w, b, X, Y)
cond1 = np.allclose(grads["dw"], np.array([[0.99845601],
                                           [2.39507239]]))
cond2 = grads["db"].round(4) == 0.0015
cond3 = cost.round(4) == 5.8015
cond = cond1 & cond2 & cond3
if cond:
    print("Test propagate is OK")
else:
    print("Test propagate FAILS")

# optimize
params, grads, costs = optimize(w, b, X, Y, num_iterations=100, lr=0.009, print_cost=False)
cond1 = np.allclose(params["w"], np.array([[0.19033591],
                                          [0.12259159]]))
cond2 = params["b"].round(4) == 1.9254
cond3 = np.allclose(grads["dw"], np.array([[0.67752042],
                                           [1.41625495]]))
cond4 = grads["db"].round(4) == 0.2192
cond = cond1 & cond2 & cond3 & cond4
if cond:
    print("Test optimize is OK")
else:
    print("Test optimize FAILS")

# predict
w = np.array([[0.1124579],
              [0.23106775]])
b = -0.3
X = np.array([[1., -1.1, -3.2],
              [1.2, 2., 0.1]])
cond = np.mean(predict(w, b, X)).round(2) == 0.67
if cond:
    print("Test predict is OK")
else:
    print("Test predict FAILS")