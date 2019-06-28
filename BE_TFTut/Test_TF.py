import numpy as np

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