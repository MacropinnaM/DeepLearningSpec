import numpy as np

from BC_Reg.backward_propagation_with_dropout import backward_propagation_with_dropout
from BC_Reg.backward_propagation_with_regularization import backward_propagation_with_regularization
from BC_Reg.compute_cost_with_regularization import compute_cost_with_regularization
from BC_Reg.forward_propagation_with_dropout import forward_propagation_with_dropout

np.random.seed(8)

# compute_cost_with_regularization
Y_assess = np.array([[1, 1, 0, 1, 0]])
W1, b1 = np.random.randn(2, 3), np.random.randn(2, 1)
W2, b2 = np.random.randn(3, 2), np.random.randn(3, 1)
W3, b3 = np.random.randn(1, 3), np.random.randn(1, 1)
parameters = {"W1": W1, "b1": b1,
              "W2": W2, "b2": b2,
              "W3": W3, "b3": b3}
A3 = np.array([[0.40682402, 0.01629284, 0.16722898, 0.10118111, 0.40682402]])
cost = compute_cost_with_regularization(A3, Y_assess, parameters, lambd=0.1)
cond = cost.round(4) == 1.8671
if cond:
    print("Test compute_cost_with_regularization is OK")
else:
    print("Test compute_cost_with_regularization FAILS")


# backward_propagation_with_regularization
X_assess = np.random.randn(3, 5)
Y_assess = np.array([[1, 1, 0, 1, 0]])
cache = (np.array([[-1.52855314, 3.32524635, 2.13994541, 2.60700654, -0.75942115],
                   [-1.98043538, 4.1600994, 0.79051021, 1.46493512, -0.45506242]]),
         np.array([[0., 0., 4.27989081, 5.21401307, 0.],
                   [0., 8.32019881, 1.58102041, 2.92987024, 0.]]),
         np.array([[-1.09989127, -0.17242821, -0.87785842],
                   [0.04221375, 0.58281521, -1.10061918]]),
         np.array([[1.14472371],
                   [0.90159072]]),
         np.array([[0.53035547, 8.02565606, 4.10524802, 5.78975856, 0.53035547],
                   [-0.69166075, -1.71413186, -3.81223329, -4.61667916, -0.69166075],
                   [-0.39675353, -2.62563561, -4.82528105, -6.0607449, -0.39675353]]),
         np.array([[1.06071093, 0., 8.21049603, 0., 1.06071093],
                   [0., 0., 0.,  0., 0.],
                   [0., 0., 0.,  0., 0.]]),
         np.array([[ 0.50249434, 0.90085595],
                   [-0.68372786, -0.12289023],
                   [-0.93576943, -0.26788808]]),
         np.array([[0.53035547],
                   [-0.69166075],
                   [-0.39675353]]),
         np.array([[-0.7415562, -0.0126646, -5.65469333, -0.0126646, -0.7415562]]),
         np.array([[0.32266394, 0.49683389, 0.00348883, 0.49683389, 0.32266394]]),
         np.array([[-0.6871727, -0.84520564, -0.67124613]]),
         np.array([[-0.0126646]]))
grads = backward_propagation_with_regularization(X_assess, Y_assess, cache, lambd = 0.7)
cond1 = np.mean(grads["dW1"]).round(4) == -0.0614
cond2 = np.mean(grads["dW2"]).round(4) == -0.0146
cond3 = np.mean(grads["dW3"]).round(4) == -0.126

cond = cond1 & cond2 & cond3
if cond:
    print("Test backward_propagation_with_regularization is OK")
else:
    print("Test backward_propagation_with_regularization FAILS")

# forward_propagation_with_dropout
X_assess = np.random.randn(3, 5)
W1, b1 = np.random.randn(2, 3), np.random.randn(2, 1)
W2, b2 = np.random.randn(3, 2), np.random.randn(3, 1)
W3, b3 = np.random.randn(1, 3), np.random.randn(1, 1)
parameters = {"W1": W1, "b1": b1,
              "W2": W2, "b2": b2,
              "W3": W3, "b3": b3}
A3, cache = forward_propagation_with_dropout(X_assess, parameters, keep_prob=0.7)
cond = np.mean(A3).round(4) == .8903
if cond:
    print("Test forward_propagation_with_dropout is OK")
else:
    print("Test forward_propagation_with_dropout FAILS")


# backward_propagation_with_dropout
X_assess = np.random.randn(3, 5)
Y_assess = np.array([[1, 1, 0, 1, 0]])
cache = (np.array([[-1.52855314, 3.32524635, 2.13994541, 2.60700654, -0.75942115],
                   [-1.98043538, 4.1600994, 0.79051021, 1.46493512, -0.45506242]]),
         np.array([[True, False, True, True, True],
                   [True, True, True, True, False]], dtype=bool),
         np.array([[0., 0., 4.27989081, 5.21401307, 0.],
                   [0., 8.32019881, 1.58102041, 2.92987024, 0.]]),
         np.array([[-1.09989127, -0.17242821, -0.87785842],
                   [0.04221375, 0.58281521, -1.10061918]]),
         np.array([[1.14472371],
                   [0.90159072]]),
         np.array([[0.53035547, 8.02565606, 4.10524802, 5.78975856, 0.53035547],
                   [-0.69166075, -1.71413186, -3.81223329, -4.61667916, -0.69166075],
                   [-0.39675353, -2.62563561, -4.82528105, -6.0607449, -0.39675353]]),
         np.array([[True, False, True, False, True],
                   [False, True, False, True, True],
                   [False, False, True, False, False]], dtype=bool),
         np.array([[1.06071093, 0., 8.21049603, 0., 1.06071093],
                   [0., 0., 0.,  0., 0.],
                   [0., 0., 0.,  0., 0.]]),
         np.array([[ 0.50249434, 0.90085595],
                   [-0.68372786, -0.12289023],
                   [-0.93576943, -0.26788808]]),
         np.array([[0.53035547],
                   [-0.69166075],
                   [-0.39675353]]),
         np.array([[-0.7415562, -0.0126646, -5.65469333, -0.0126646, -0.7415562]]),
         np.array([[0.32266394, 0.49683389, 0.00348883, 0.49683389, 0.32266394]]),
         np.array([[-0.6871727, -0.84520564, -0.67124613]]),
         np.array([[-0.0126646]]))

gradients = backward_propagation_with_dropout(X_assess, Y_assess, cache, keep_prob=0.8)
cond1 = np.mean(gradients["dA1"]).round(4) == .0841
cond2 = np.mean(gradients["dA2"]).round(4) == .0681
cond = cond1 & cond2
if cond:
    print("Test backward_propagation_with_dropout is OK")
else:
    print("Test backward_propagation_with_dropout FAILS")


