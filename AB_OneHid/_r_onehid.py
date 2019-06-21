import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model

from AB_OneHid._m_onehid import onehid
from AB_OneHid.load_flower_dataset import load_flower_dataset
from AB_OneHid.plot_decision_boundary import plot_decision_boundary
from AB_OneHid.predict import predict

# Load dataset
X, Y = load_flower_dataset(n_samples=400, n_classes=2, dim=2, max_ray=4)

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=40, cmap=plt.cm.Spectral);
plt.show()

shape_X, shape_Y = X.shape, Y.shape
m = X.size / 2  # training set size

print('The shape of X is: ' + str(shape_X))
print('The shape of Y is: ' + str(shape_Y))
print('I have m = %d training examples!' % (m))


### SKLEARN LOGISTIC REGRESSION ###
print('\n' + "\033[1m" + " SKLEARN LOGISTIC REGRESSION:" + "\033[0m")
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);

plot_decision_boundary(lambda x: clf.predict(x), X, Y.ravel(), h=0.01)
plt.title("Logistic Regression")

LR_predictions = clf.predict(X.T)
print('Accuracy of logistic regression: %d ' % float((np.dot(Y, LR_predictions) +
                                                      np.dot(1-Y, 1-LR_predictions))/float(Y.size)*100) +
      "% (percentage of correctly labelled datapoints)")


### ONE HIDEEN LAYER NEURAL NETWORK ###
print('\n' + "\033[1m" + " ONE HIDEEN LAYER NEURAL NETWORK:" + "\033[0m")
parameters = onehid(X, Y, n_h=4, num_iterations=10000, print_cost=True)

plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y, h=0.01)
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(parameters, X)
print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100) + '%')


### TUNING HIDDEN LAYER SIZE ###
print('\n' + "\033[1m" + " TUNING HIDDEN LAYER SIZE:" + "\033[0m")
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = onehid(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y, h=0.01)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100)
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))

