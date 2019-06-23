import numpy as np

from AB_OneHid._m_onehid import onehid
from AB_OneHid.backward_propagation import backward_propagation
from AB_OneHid.compute_cost import compute_cost
from AB_OneHid.forward_propagation import forward_propagation
from AB_OneHid.predict import predict
from AB_OneHid.update_parameters import update_parameters
from AB_OneHid.initialize_parameters import initialize_parameters
from AB_OneHid.layer_sizes import layer_sizes

np.random.seed(8)


# layer_sizes
X_assess, Y_assess = np.random.randn(5, 3), np.random.randn(2, 3)
n_x, n_h, n_y = layer_sizes(X_assess, Y_assess)
cond1 = n_x == 5
cond2 = n_h == 4
cond3 = n_y == 2
cond = cond1 & cond2 & cond3
if cond:
    print("Test layer_sizes is OK")
else:
    print("Test layer_sizes FAILS")


# initialize_parameters
n_x, n_h, n_y = 2, 4, 1
parameters = initialize_parameters(n_x, n_h, n_y)
cond1 = np.mean(parameters["W1"]).round(4) == .0024
cond2 = np.mean(parameters["b1"]).round(4) == .0063
cond3 = np.mean(parameters["W2"]).round(4) == -0.0074
cond4 = np.mean(parameters["b2"]).round(4) == .0086
cond = cond1 & cond2 & cond3 & cond4
if cond:
    print("Test initialize_parameters is OK")
else:
    print("Test initialize_parameters FAILS")


# forward_propagation
X_assess = np.random.randn(2, 3)
parameters = {'W1': np.array([[-0.00416758, -0.00056267],
                              [-0.02136196, 0.01640271],
                              [-0.01793436, -0.00841747],
                              [0.00502881, -0.01245288]]),
              'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
              'b1': np.random.randn(4, 1),
              'b2': np.array([[-1.3]])}
A2, cache = forward_propagation(X_assess, parameters)
cond1 = np.mean(cache['Z1']).round(4) == -0.2911
cond2 = np.mean(cache['A1']).round(4) == -0.1218
cond3 = np.mean(cache['Z2']).round(4) == -1.2871
cond4 = np.mean(cache['A2']).round(4) == .2163
cond = cond1 & cond2 & cond3 & cond4
if cond:
    print("Test initialize_parameters is OK")
else:
    print("Test initialize_parameters FAILS")


# compute_cost
Y_assess = np.random.randn(1, 3) > 0
A2 = (np.array([[0.5002307, 0.49985831, 0.50023963]]))
cost = compute_cost(A2, Y_assess)
cond = cost.round(4) == .6932
if cond:
    print("Test compute_cost is OK")
else:
    print("Test compute_cost FAILS")


# backward_propagation
parameters = {'W1': np.array([[-0.00416758, -0.00056267],
                              [-0.02136196, 0.01640271],
                              [-0.01793436, -0.00841747],
                              [ 0.00502881, -0.01245288]]),
              'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
              'b1': np.random.randn(4, 1),
              'b2': np.array([[-1.3]])}
X_assess = np.random.randn(2, 3)
Y_assess = (np.random.randn(1, 3) > 0)
parameters = {'W1': np.array([[-0.00416758, -0.00056267],
                              [-0.02136196, 0.01640271],
                              [-0.01793436, -0.00841747],
                              [0.00502881, -0.01245288]]),
              'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
              'b1': np.random.randn(4, 1),
              'b2': np.array([[-1.3]])}

cache = {'A1': np.array([[-0.00616578, 0.0020626, 0.00349619],
                         [-0.05225116, 0.02725659, -0.02646251],
                         [-0.02009721, 0.0036869, 0.02883756],
                         [0.02152675, -0.01385234, 0.02599885]]),
         'A2': np.array([[0.5002307, 0.49985831, 0.50023963]]),
         'Z1': np.array([[-0.00616586, 0.0020626, 0.0034962],
                         [-0.05229879, 0.02726335, -0.02646869],
                         [-0.02009991, 0.00368692, 0.02884556],
                         [0.02153007, -0.01385322, 0.02600471]]),
         'Z2': np.array([[0.00092281, -0.00056678, 0.00095853]])}
grads = backward_propagation(parameters, cache, X_assess, Y_assess)
cond1 = np.mean(grads["dW1"]).round(4) == -0.0
cond2 = np.mean(grads["db1"]).round(4) == -0.0004
cond3 = np.mean(grads["dW2"]).round(4) == -0.0045
cond4 = np.mean(grads["db2"]).round(4) == -0.1666
cond = cond1 & cond2 & cond3 & cond4
if cond:
    print("Test backward_propagation is OK")
else:
    print("Test backward_propagation FAILS")


# update_parameters
parameters = {'W1': np.array([[-0.00416758, -0.00056267],
                              [-0.02136196, 0.01640271],
                              [-0.01793436, -0.00841747],
                              [ 0.00502881, -0.01245288]]),
              'W2': np.array([[-0.01057952, -0.00909008, 0.00551454, 0.02292208]]),
              'b1': np.random.randn(4, 1),
              'b2': np.array([[-1.3]])}
grads = {'dW1': np.array([[0.00023322, -0.00205423],
                          [0.00082222, -0.00700776],
                          [-0.00031831, 0.0028636],
                          [-0.00092857, 0.00809933]]),
         'dW2': np.array([[-1.75740039e-05, 3.70231337e-03, -1.25683095e-03, -2.55715317e-03]]),
         'db1': np.array([[1.05570087e-07],
                  [-3.81814487e-06],
                  [-1.90155145e-07],
                  [5.46467802e-07]]),
         'db2': np.array([[-1.08923140e-05]])}
parameters = update_parameters(parameters, grads)
cond1 = np.mean(parameters["W1"]).round(4) == -0.0057
cond2 = np.mean(parameters["b1"]).round(4) == -0.7148
cond3 = np.mean(parameters["W2"]).round(4) == .0022
cond4 = np.mean(parameters["b2"]).round(4) == -1.3
cond = cond1 & cond2 & cond3 & cond4
if cond:
    print("Test update_parameters is OK")
else:
    print("Test update_parameters FAILS")


# onehid
X_assess, Y_assess = np.random.randn(2, 3), (np.random.randn(1, 3) > 0)
parameters = onehid(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
cond1 = np.mean(parameters["W1"]).round(4) == .0383
cond2 = np.mean(parameters["b1"]).round(4) == -1.4633
cond3 = np.mean(parameters["W2"]).round(4) == -2.9887
cond4 = np.mean(parameters["b2"]).round(4) == 1.145
cond = cond1 & cond2 & cond3 & cond4
if cond:
    print("Test onehid is OK")
else:
    print("Test onehid FAILS")


# predict
X_assess = np.random.randn(2, 3)
parameters = {'W1': np.array([[-0.00615039, 0.0169021],
                              [-0.02311792, 0.03137121],
                              [-0.0169217, -0.01752545],
                              [0.00935436, -0.05018221]]),
              'W2': np.array([[-0.0104319, -0.04019007, 0.01607211, 0.04440255]]),
              'b1': np.array([[-8.97523455e-07],
                              [8.15562092e-06],
                              [6.04810633e-07],
                              [-2.54560700e-06]]),
              'b2': np.array([[9.14954378e-05]])}
predictions = predict(parameters, X_assess)
cond = np.mean(predictions).round(4) == 0.6667
if cond:
    print("Test predict is OK")
else:
    print("Test predict FAILS")




"""
print(np.mean(___).round(4))

"""