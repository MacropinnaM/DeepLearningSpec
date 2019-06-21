import numpy as np

from AC_DeepNet.compute_cost import compute_cost
from AC_DeepNet.initialize_parameters_deep import initialize_parameters_deep
from AC_DeepNet.l_model_backward import l_model_backward
from AC_DeepNet.l_model_forward import l_model_forward
from AC_DeepNet.linear_activation_backward import linear_activation_backward
from AC_DeepNet.linear_activation_forward import linear_activation_forward
from AC_DeepNet.linear_backward import linear_backward
from AC_DeepNet.linear_forward import linear_forward
from AC_DeepNet.update_parameters import update_parameters

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
cond = np.allclose(A, [[0.41367206, 0.78630206]])
if cond:
    print("Test linear_activation_forward with sigmoid is OK")
else:
    print("Test linear_activation_forward with sigmoid FAILS")
A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
cond = np.allclose(A, [[0., 1.30277749]])
if cond:
    print("Test linear_activation_forward with relu is OK")
else:
    print("Test linear_activation_forward with relu FAILS")


# l_model_forward
X = np.random.randn(4, 2)
W1 = np.random.randn(3, 4)
b1 = np.random.randn(3, 1)
W2 = np.random.randn(1, 3)
b2 = np.random.randn(1, 1)
parameters = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2}

AL, caches = l_model_forward(X, parameters)
cond1 = np.mean(AL).round(4) == .5207
cond2 = len(caches) == 2
cond = cond1 & cond2
if cond:
    print("Test l_model_forward is OK")
else:
    print("Test l_model_forward FAILS")


# compute_cost
Y = np.asarray([[1, 1, 1]])
AL = np.array([[.8, .9, .4]])

cost = compute_cost(AL, Y)
cond = cost.round(4) == .4149
if cond:
    print("Test compute_cost is OK")
else:
    print("Test compute_cost FAILS")


# linear_backward
dZ = np.random.randn(1, 2)
A = np.random.randn(3, 2)
W = np.random.randn(1, 3)
b = np.random.randn(1, 1)
linear_cache = (A, W, b)

dA_prev, dW, db = linear_backward(dZ, linear_cache)
cond1 = np.mean(dA_prev).round(4) == -.0035
cond2 = np.mean(dW).round(4) == .0301
cond3 = np.mean(db).round(4) == -.0156
cond = cond1 & cond2 & cond3
if cond:
    print("Test linear_backward is OK")
else:
    print("Test linear_backward FAILS")


# linear_activation_backward
dAL = np.random.randn(1, 2)
A = np.random.randn(3, 2)
W = np.random.randn(1, 3)
b = np.random.randn(1, 1)
Z = np.random.randn(1, 2)
linear_cache = (A, W, b)
activation_cache = Z
linear_activation_cache = (linear_cache, activation_cache)

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation="sigmoid")
cond1 = np.mean(dA_prev).round(4) == .0635
cond2 = np.mean(dW).round(4) == .033
cond3 = np.mean(db).round(4) == -.1084
cond = cond1 & cond2 & cond3
if cond:
    print("Test linear_activation_backward with sigmoid is OK")
else:
    print("Test linear_activation_backward with sigmoid FAILS")

dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation="relu")
cond1 = np.mean(dA_prev).round(4) == -.0445
cond2 = np.mean(dW).round(4) == .0266
cond3 = np.mean(db).round(4) == .076
cond = cond1 & cond2 & cond3
if cond:
    print("Test linear_activation_backward with relu is OK")
else:
    print("Test linear_activation_backward with relu FAILS")


# l_model_backward
AL = np.random.randn(1, 2)
Y_assess = np.array([[1, 0]])
A1 = np.random.randn(4, 2)
W1 = np.random.randn(3, 4)
b1 = np.random.randn(3, 1)
Z1 = np.random.randn(3, 2)
linear_cache_activation_1 = ((A1, W1, b1), Z1)
A2 = np.random.randn(3, 2)
W2 = np.random.randn(1, 3)
b2 = np.random.randn(1, 1)
Z2 = np.random.randn(1, 2)
linear_cache_activation_2 = ((A2, W2, b2), Z2)
caches = (linear_cache_activation_1, linear_cache_activation_2)

grads = l_model_backward(AL, Y_assess, caches)
cond1 = np.mean(grads["dW1"]).round(4) == -.0268
cond2 = np.mean(grads["db1"]).round(4) == -.1046
cond3 = np.mean(grads["dA1"]).round(4) == -.0032
cond = cond1 & cond2 & cond3
if cond:
    print("Test l_model_backward is OK")
else:
    print("Test l_model_backward FAILS")


# update_parameters
np.random.seed(4)
W1 = np.random.randn(3, 4)
b1 = np.random.randn(3, 1)
W2 = np.random.randn(1, 3)
b2 = np.random.randn(1, 1)
parameters = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2}
np.random.seed(8)
dW1 = np.random.randn(3, 4)
db1 = np.random.randn(3, 1)
dW2 = np.random.randn(1, 3)
db2 = np.random.randn(1, 1)
grads = {"dW1": dW1,
         "db1": db1,
         "dW2": dW2,
         "db2": db2}

parameters = update_parameters(parameters, grads, lr=0.1)
cond1 = np.mean(W1).round(4) == -.174
cond2 = np.mean(b1).round(4) == -.1332
cond3 = np.mean(W2).round(4) == .4304
cond4 = np.mean(b2).round(4) == .7233
cond = cond1 & cond2 & cond3 & cond4
if cond:
    print("Test update_parameters is OK")
else:
    print("Test update_parameters FAILS")
