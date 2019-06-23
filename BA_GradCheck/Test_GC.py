import numpy as np

from BA_GradCheck.backward_propagation import backward_propagation
from BA_GradCheck.backward_propagation_n import backward_propagation_n
from BA_GradCheck.forward_propagation import forward_propagation
from BA_GradCheck.forward_propagation_n import forward_propagation_n
from BA_GradCheck.gradient_check import gradient_check
from BA_GradCheck.gradient_check_n import gradient_check_n

np.random.seed(8)

# forward_propagation
x, theta = 2, 4
J = forward_propagation(x, theta)
cond = J == 8
if cond:
    print("Test forward_propagation is OK")
else:
    print("Test forward_propagation FAILS")


# backward_propagation
x, theta = 2, 4
dtheta = backward_propagation(x, theta)
cond = dtheta == 2
if cond:
    print("Test backward_propagation is OK")
else:
    print("Test backward_propagation FAILS")


# gradient_check
x, theta = 2, 4
difference = gradient_check(x, theta)
cond = difference.round(12) == 2.92e-10
if cond:
    print("Test backward_propagation is OK")
else:
    print("Test backward_propagation FAILS")


# forward_propagation_n
# backward_propagation_n
