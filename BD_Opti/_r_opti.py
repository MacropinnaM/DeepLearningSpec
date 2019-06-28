import matplotlib.pyplot as plt

from BD_Opti._m_opti import model
from BD_Opti.load_moons import load_moons
from BD_Opti.plot_decision_boundary import plot_decision_boundary
from BD_Opti.predict import predict


# Load data
from BD_Opti.predict_dec import predict_dec

n_samples = 300
noise = 0.2
train_X, train_Y = load_moons(n_samples, noise)

### MINI-BATCH GRADIENT DESCENT ###
print('\n' + "\033[1m" + " MINI-BATCH GRADIENT DESCENT:" + "\033[0m")
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer="gd")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


### MINI-BATCH GRADIENT DESCENT WITH MOMENTUM ###
print('\n' + "\033[1m" + " MINI-BATCH GRADIENT DESCENT WITH MOMENTUM:" + "\033[0m")
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, beta=0.9, optimizer="momentum")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


### MINI-BATCH WITH ADAM MODE ###
print('\n' + "\033[1m" + " MINI-BATCH WITH ADAM MODE:" + "\033[0m")
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer="adam")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)