import numpy as np
import tensorflow as tf

from BE_TFTut.compute_cost import compute_cost
from BE_TFTut.create_placeholders import create_placeholders
from BE_TFTut.forward_propagation import forward_propagation
from BE_TFTut.initialize_parameters import initialize_parameters
from BE_TFTut.linear_function import linear_function
from BE_TFTut.cost import cost


# linear_function
from BE_TFTut.one_hot_matrix import one_hot_matrix
from BE_TFTut.ones import ones
from BE_TFTut.sigmoid import sigmoid

result = linear_function()
cond = np.mean(result).round(4) == -0.2831
if cond:
    print("Test linear_function is OK")
else:
    print("Test linear_function FAILS")


# sigmoid
result1 = sigmoid(0)
result2 = sigmoid(12)
cond1 = result1 == 0.5
cond2 = result2.round(4) == 1
cond = cond1 & cond2
if cond:
    print("Test sigmoid is OK")
else:
    print("Test sigmoid FAILS")


# cost
logits = sigmoid(np.array([0.2, 0.4, 0.7, 0.9]))
cost = cost(logits, np.array([0, 0, 1, 1]))
cond = np.allclose(cost, np.array([1.0053872, 1.0366409, 0.41385433, 0.39956614]))
if cond:
    print("Test cost is OK")
else:
    print("Test cost FAILS")


# one_hot_matrix
labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels, C=4)
cond = np.mean(one_hot) == 0.25
if cond:
    print("Test one_hot_matrix is OK")
else:
    print("Test one_hot_matrix FAILS")


# ones
result = ones([3])
cond = result1 * 3 == 3
cond = cond1 & cond2
if cond:
    print("Test ones is OK")
else:
    print("Test ones FAILS")


# create_placeholders
n_x, n_y = 12288, 6
X, Y = create_placeholders(n_x, n_y)
cond1 = X.shape[0] == n_x
cond2 = Y.shape[0] == n_y
cond = cond1 & cond2
if cond:
    print("Test create_placeholders is OK")
else:
    print("Test create_placeholders FAILS")


# initialize_parameters
tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
cond1 = parameters["W1"].shape == (25, 12288)
cond2 = parameters["b1"].shape == (25, 1)
cond3 = parameters["W2"].shape == (12, 25)
cond4 = parameters["b2"].shape == (12, 1)
cond = cond1 & cond2 & cond3 & cond4
if cond:
    print("Test initialize_parameters is OK")
else:
    print("Test initialize_parameters FAILS")


# forward_propagation
tf.reset_default_graph()
with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
cond = Z3.shape[0] == 6
if cond:
    print("Test forward_propagation is OK")
else:
    print("Test forward_propagation FAILS")


# compute_cost
tf.reset_default_graph()
with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
cond = cost.shape == ()
if cond:
    print("Test compute_cost is OK")
else:
    print("Test compute_cost FAILS")