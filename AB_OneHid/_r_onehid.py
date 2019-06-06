import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model

from AB_OneHid._m_onehid import onehid
from AB_OneHid.load_planar_dataset import load_planar_dataset
from AB_OneHid.plot_decision_boundary import plot_decision_boundary
from AB_OneHid.predict import predict

X, Y = load_planar_dataset()
# Visualize the data:
plt.scatter(X[0, :], X[1, :], c=Y.ravel(), s=40, cmap=plt.cm.Spectral);
plt.show()

shape_X = X.shape
shape_Y = X.shape
m = X.size / 2  # training set size

print('The shape of X is: ' + str(shape_X))
print('The shape of Y is: ' + str(shape_Y))
print('I have m = %d training examples!' % (m))

# Simple logistic regression
print("\033[1m" + "SKLearn logistic regression:" + "\033[0m")
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y.T);
# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y.ravel())
plt.title("Logistic Regression")

# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")

# Build a model with a n_h-dimensional hidden layer
parameters = onehid(X, Y, n_h=4, num_iterations=10000, print_cost=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))