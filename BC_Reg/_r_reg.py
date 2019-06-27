import matplotlib.pyplot as plt

from BC_Reg._m_reg import model
from BC_Reg.load_2D_dataset import load_2D_dataset
from BC_Reg.plot_decision_boundary import plot_decision_boundary
from BC_Reg.predict import predict
from BC_Reg.predict_dec import predict_dec

# Load dataset
comp = 'iKosh'
file_name = 'data.mat'
train_X, train_Y, test_X, test_Y = load_2D_dataset(comp, file_name)


### NON-REGULARIZED MODEL ###
print('\n' + "\033[1m" + " NON-REGULARIZED MODEL:" + "\033[0m")
parameters = model(train_X, train_Y)
print("On the training set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


### L2 REGULARIZATION ###
print('\n' + "\033[1m" + " L2 REGULARIZATION:" + "\033[0m")
parameters = model(train_X, train_Y, lambd = 0.7)
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)


### DROPOUT ###
print('\n' + "\033[1m" + " DROPOUT:" + "\033[0m")
parameters = model(train_X, train_Y, keep_prob = 0.86, lr = 0.3)
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)