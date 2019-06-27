import numpy as np

from BB_Ini.initialize_parameters_he import initialize_parameters_he
from BB_Ini.initialize_parameters_random import initialize_parameters_random
from BB_Ini.initialize_parameters_xavier import initialize_parameters_xavier
from BB_Ini.initialize_parameters_zeros import initialize_parameters_zeros


# initialize_parameters_zeros
layers_dims = [3, 2, 1]
parameters = initialize_parameters_zeros(layers_dims)
cond1 = np.mean(parameters["W1"]) == 0
cond2 = np.mean(parameters["b1"]) == 0
cond3 = np.mean(parameters["W2"]) == 0
cond4 = np.mean(parameters["b2"]) == 0
cond = cond1 & cond2 & cond3 & cond4
if cond:
    print("Test initialize_parameters_zeros is OK")
else:
    print("Test initialize_parameters_zeros FAILS")


# initialize_parameters_random
layers_dims = [3, 2, 1]
parameters = initialize_parameters_random([3, 2, 1])
cond1 = np.mean(parameters["W1"]).round(4) == -3.3958
cond2 = np.mean(parameters["b1"]).round(4) == 0.0
cond3 = np.mean(parameters["W2"]).round(4) == 19.662
cond4 = np.mean(parameters["b2"]).round(4) == 0.0
cond = cond1 & cond2 & cond3 & cond4
if cond:
    print("Test initialize_parameters_random is OK")
else:
    print("Test initialize_parameters_random FAILS")


# initialize_parameters_he
layers_dims = [3, 2, 1]
parameters = initialize_parameters_he([3, 2, 1])
cond1 = np.mean(parameters["W1"]).round(4) == -0.2773
cond2 = np.mean(parameters["b1"]).round(4) == 0.0
cond3 = np.mean(parameters["W2"]).round(4) == 1.9662
cond4 = np.mean(parameters["b2"]).round(4) == 0.0
cond = cond1 & cond2 & cond3 & cond4
if cond:
    print("Test initialize_parameters_he is OK")
else:
    print("Test initialize_parameters_he FAILS")

# initialize_parameters_xavier
layers_dims = [3, 2, 1]
parameters = initialize_parameters_xavier([3, 2, 1])
cond1 = np.mean(parameters["W1"]).round(4) == -0.1961
cond2 = np.mean(parameters["b1"]).round(4) == 0.0
cond3 = np.mean(parameters["W2"]).round(4) == 1.3903
cond4 = np.mean(parameters["b2"]).round(4) == 0.0
cond = cond1 & cond2 & cond3 & cond4
if cond:
    print("Test initialize_parameters_xavier is OK")
else:
    print("Test initialize_parameters_xavier FAILS")
