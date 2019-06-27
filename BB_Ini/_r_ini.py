import matplotlib.pyplot as plt

from BB_Ini._m_ini import model
from BB_Ini.load_circles import load_circles

# Load dataset
from BB_Ini.plot_decision_boundary import plot_decision_boundary
from BB_Ini.predict import predict
from BB_Ini.predict_dec import predict_dec

n_samles_1 = 300
n_samples_2 = 100
noise = 0.05
train_X, train_Y, test_X, test_Y = load_circles(n_samles_1, n_samples_2, noise)


### ZERO INITIALIZATION ###
print('\n' + "\033[1m" + " ZERO INITIALIZATION:" + "\033[0m")
parameters = model(train_X, train_Y, ini="zeros")
print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

print("predictions_train = " + str(predictions_train))
print("predictions_test = " + str(predictions_test))

plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


### RANDOM INITIALIZATION ###
print('\n' + "\033[1m" + " RANDOM INITIALIZATIONN:" + "\033[0m")
parameters = model(train_X, train_Y, ini="random")
print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

print(predictions_train)
print(predictions_test)

plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


### HE INITIALIZATION ###
print('\n' + "\033[1m" + " HE INITIALIZATIONN:" + "\033[0m")
parameters = model(train_X, train_Y, ini="he")
print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

print(predictions_train)
print(predictions_test)

plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


### XAVIER INITIALIZATION ###
print('\n' + "\033[1m" + " XAVIER INITIALIZATIONN:" + "\033[0m")
parameters = model(train_X, train_Y, ini="xavier")
print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

print(predictions_train)
print(predictions_test)

plt.title("Model with Xavier initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


