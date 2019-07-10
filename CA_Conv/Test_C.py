import numpy as np
import tensorflow as tf

from CA_Conv.compute_cost import compute_cost
from CA_Conv.conv_forward import conv_forward
from CA_Conv.conv_single_step import conv_single_step
from CA_Conv.create_placeholders import create_placeholders
from CA_Conv.forward_propagation import forward_propagation
from CA_Conv.initialize_parameters import initialize_parameters
from CA_Conv.pool_forward import pool_forward
from CA_Conv.zero_pad import zero_pad

np.random.seed(8)


# zero_pad
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)
cond1 = np.sum(x_pad.shape) == 20
cond2 = np.mean(x[1, 1]).round(4) == -.1565
cond3 = np.mean(x_pad[1, 1]) == .0
cond = cond1 & cond2 & cond3
if cond:
    print("Test zero_pad is OK")
else:
    print("Test zero_pad FAILS")


# conv_single_step
a_slice_prev = np.random.randn(4, 4, 3)
W, b = np.random.randn(4, 4, 3), np.random.randn(1, 1, 1)
Z = conv_single_step(a_slice_prev, W, b)
cond = np.round(Z, 4) == -6.3872
if cond:
    print("Test conv_single_step is OK")
else:
    print("Test conv_single_step FAILS")


# conv_forward
A_prev = np.random.randn(10,4,4,3)
W, b = np.random.randn(2, 2, 3, 8), np.random.randn(1, 1, 1, 8)
hparameters = {"pad": 2,
               "stride": 2}
Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
cond1 = np.mean(Z).round(4) == 0.2246
cond2 = np.mean(Z[3, 2, 1]).round(4) == -0.0223
cond3 = np.mean(cache_conv[0][1][2][3]).round(4) == -0.4141
cond = cond1 & cond2 & cond3
if cond:
    print("Test conv_forward is OK")
else:
    print("Test conv_forward FAILS")


# pool_forward
A_prev = np.random.randn(2, 4, 4, 3)
hparameters = {"stride" : 2, "f": 3}
A_max, cache = pool_forward(A_prev, hparameters, mode="max")
cond = np.mean(A_max).round(4) == 1.2559
if cond:
    print("Test pool_forward on max mode is OK")
else:
    print("Test pool_forward on max mode FAILS")
A_aver, cache = pool_forward(A_prev, hparameters, mode="average")
cond = np.mean(A_aver).round(4) == -0.1075
if cond:
    print("Test pool_forward on average mode is OK")
else:
    print("Test pool_forward on average mode FAILS")


# create_placeholders
X, Y = create_placeholders(64, 64, 3, 6)
cond1 = np.sum(X.shape[1:]) == 131
cond2 = np.sum(Y.shape[1:]) == 6
cond = cond1 & cond2
if cond:
    print("Test create_placeholders is OK")
else:
    print("Test create_placeholders FAILS")


# initialize_parameters
tf.reset_default_graph()
with tf.Session() as sess_test:
    parameters = initialize_parameters()
    init = tf.global_variables_initializer()
    sess_test.run(init)
cond1 = np.sum(parameters["W1"].shape) == 19
cond2 = np.sum(parameters["W2"].shape) == 28
cond = cond1 & cond2
if cond:
    print("Test initialize_parameters is OK")
else:
    print("Test initialize_parameters FAILS")


# forward_propagation
tf.reset_default_graph()
with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z3, {X: np.random.randn(2, 64, 64, 3), Y: np.random.randn(2, 6)})
cond = np.allclose([[1.9580412, -3.3216517, 2.8901303, -0.7775271, -0.5990415, -0.22340764],
                  [1.6749594, -3.1823099, 3.181048, -0.56886154, -0.6549634, -0.19792166]], a)
if cond:
    print("Test forward_propagation is OK")
else:
    print("Test forward_propagation FAILS")


# compute_cost
tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(1)
    X, Y = create_placeholders(64, 64, 3, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(cost, {X: np.random.randn(4,64,64,3), Y: np.random.randn(4,6)})
cond = int(a) == 6
if cond:
    print("Test compute_cost is OK")
else:
    print("Test compute_cost FAILS")


