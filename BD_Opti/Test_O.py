import numpy as np

from BD_Opti.initialize_adam import initialize_adam
from BD_Opti.initialize_velocity import initialize_velocity
from BD_Opti.random_mini_batches import random_mini_batches
from BD_Opti.update_parameters_with_adam import update_parameters_with_adam
from BD_Opti.update_parameters_with_gd import update_parameters_with_gd
from BD_Opti.update_parameters_with_momentum import update_parameters_with_momentum

np.random.seed(8)


#update_parameters_with_gd
lr = 0.01
W1, b1 = np.random.randn(2, 3), np.random.randn(2, 1)
W2, b2 = np.random.randn(3, 3), np.random.randn(3, 1)
parameters = {"W1": W1, "b1": b1,
              "W2": W2, "b2": b2}
dW1, db1 = np.random.randn(2, 3), np.random.randn(2, 1)
dW2, db2 = np.random.randn(3, 3), np.random.randn(3, 1)
grads = {"dW1": dW1, "db1": db1,
         "dW2": dW2, "db2": db2}
parameters = update_parameters_with_gd(parameters, grads, lr)
cond1 = np.mean(parameters["W1"]).round(4) == -0.3377
cond2 = np.mean(parameters["b1"]).round(4) == 1.9577
cond3 = np.mean(parameters["W2"]).round(4) == 0.0443
cond4 = np.mean(parameters["b2"]).round(4) == -0.1614
cond = cond1 & cond2 & cond3 & cond4
if cond:
    print("Test update_parameters_with_gd is OK")
else:
    print("Test update_parameters_with_gd FAILS")


# random_mini_batches
mb_size = 64
X_assess = np.random.randn(12288, 148)
Y_assess = np.random.randn(1, 148) < 0.5
mini_batches = random_mini_batches(X_assess, Y_assess, mb_size)

cond1 = mini_batches[0][0].shape == (12288, 64)
cond2 = mini_batches[1][0].shape == (12288, 64)
cond3 = mini_batches[2][0].shape == (12288, 20)
cond3 = mini_batches[0][1].shape == (1, 64)
cond4 = mini_batches[1][1].shape == (1, 64)
cond5 = mini_batches[2][1].shape == (1, 20)
cond6 = np.mean(list(mini_batches[0][0][0][0:3])).round(4) == 0.3535
cond = cond1 & cond2 & cond3 & cond4 & cond5 & cond6
if cond:
    print("Test random_mini_batches is OK")
else:
    print("Test random_mini_batches FAILS")


# initialize_velocity
W1, b1 = np.random.randn(2, 3), np.random.randn(2,1)
W2, b2 = np.random.randn(3, 3), np.random.randn(3,1)
parameters = {"W1": W1, "b1": b1,
              "W2": W2, "b2": b2}
v = initialize_velocity(parameters)
cond1 = np.mean(v["dW1"]) == 0
cond2 = np.mean(v["db1"]) == 0
cond3 = np.mean(v["dW2"]) == 0
cond4 = np.mean(v["db2"]) == 0
cond = cond1 & cond2 & cond3 & cond4
if cond:
    print("Test initialize_velocity is OK")
else:
    print("Test initialize_velocity FAILS")


# update_parameters_with_momentum
W1, b1 = np.random.randn(2, 3), np.random.randn(2, 1)
W2, b2 = np.random.randn(3, 3), np.random.randn(3, 1)
parameters = {"W1": W1, "b1": b1,
              "W2": W2, "b2": b2}
dW1, db1 = np.random.randn(2, 3), np.random.randn(2, 1)
dW2, db2 = np.random.randn(3, 3), np.random.randn(3, 1)
grads = {"dW1": dW1, "db1": db1,
         "dW2": dW2, "db2": db2}
v = {'dW1': np.array([[0., 0., 0.],
                      [0., 0., 0.]]),
     'dW2': np.array([[0., 0., 0.],
                      [0., 0., 0.],
                      [0., 0., 0.]]),
     'db1': np.array([[0.],
                      [0.]]),
     'db2': np.array([[0.],
                      [0.],
                      [0.]])}
parameters, v = update_parameters_with_momentum(parameters, grads, v, beta=0.9, lr=0.01)
cond1 = np.mean(parameters["W1"]).round(4) == 0.246
cond2 = np.mean(parameters["b1"]).round(4) == -1.1556
cond3 = np.mean(parameters["W2"]).round(4) == -0.3521
cond4 = np.mean(parameters["b2"]).round(4) == 0.1225
cond5 = np.mean(v["dW1"]).round(4) == -0.0292
cond6 = np.mean(v["db1"]).round(4) == -0.1514
cond7 = np.mean(v["dW2"]).round(4) == -0.0246
cond8 = np.mean(v["db2"]).round(4) == 0.0207
cond = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 &cond7 & cond8
if cond:
    print("Test update_parameters_with_momentum is OK")
else:
    print("Test update_parameters_with_momentum FAILS")


# initialize_adam
W1, b1 = np.random.randn(2, 3), np.random.randn(2, 1)
W2, b2 = np.random.randn(3, 3), np.random.randn(3, 1)
parameters = {"W1": W1, "b1": b1,
              "W2": W2, "b2": b2}
v, s = initialize_adam(parameters)
cond1 = np.mean(v["dW1"]).round(4) == 0
cond2 = np.mean(v["db1"]).round(4) == 0
cond3 = np.mean(v["dW2"]).round(4) == 0
cond4 = np.mean(v["db2"]).round(4) == 0
cond5 = np.mean(s["dW1"]).round(4) == 0
cond6 = np.mean(s["db1"]).round(4) == 0
cond7 = np.mean(s["dW2"]).round(4) == 0
cond8 = np.mean(s["db2"]).round(4) == 0
cond = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 &cond7 & cond8
if cond:
    print("Test initialize_adam is OK")
else:
    print("Test initialize_adam FAILS")


# update_parameters_with_adam
v = {'dW1': np.array([[0., 0., 0.],
                      [0., 0., 0.]]),
     'dW2': np.array([[0., 0., 0.],
                      [0., 0., 0.],
                      [0., 0., 0.]]),
     'db1': np.array([[0.],
                      [0.]]),
     'db2': np.array([[0.],
                      [0.],
                      [0.]])}
s = {'dW1': np.array([[0., 0., 0.],
                      [0., 0., 0.]]),
     'dW2': np.array([[0., 0., 0.],
                      [0., 0., 0.],
                      [0., 0., 0.]]),
     'db1': np.array([[0.],
                      [0.]]),
     'db2': np.array([[0.],
                      [0.],
                      [0.]])}
W1, b1 = np.random.randn(2, 3), np.random.randn(2, 1)
W2, b2 = np.random.randn(3, 3), np.random.randn(3, 1)
parameters = {"W1": W1, "b1": b1,
              "W2": W2, "b2": b2}
dW1, db1 = np.random.randn(2, 3), np.random.randn(2, 1)
dW2, db2 = np.random.randn(3, 3), np.random.randn(3, 1)
grads = {"dW1": dW1, "db1": db1,
         "dW2": dW2, "db2": db2}
parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t=2)
cond1 = np.mean(parameters["W1"]).round(4) == 0.9772
cond2 = np.mean(parameters["b1"]).round(4) == 1.1493
cond3 = np.mean(parameters["W2"]).round(4) == -0.4317
cond4 = np.mean(parameters["b2"]).round(4) == 0.4283
cond5 = np.mean(v["dW1"]).round(4) == 0.0273
cond6 = np.mean(v["db1"]).round(4) == 0.0792
cond7 = np.mean(v["dW2"]).round(4) == 0.0038
cond8 = np.mean(v["db2"]).round(4) == 0.032
cond9 = np.mean(s["dW1"]).round(4) == 0.0004
cond10 = np.mean(s["db1"]).round(4) == 0.0017
cond11 = np.mean(s["dW2"]).round(4) == 0.0006
cond12 = np.mean(s["db2"]).round(4) == 0.0013
cond = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8 & cond9 & cond10 & cond11 & cond12
if cond:
    print("Test update_parameters_with_adam is OK")
else:
    print("Test update_parameters_with_adam FAILS")