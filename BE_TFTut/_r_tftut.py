import numpy as np
import matplotlib.pyplot as plt

from BE_TFTut._m_tftut import model
from BE_TFTut.convert_to_one_hot import convert_to_one_hot
from BE_TFTut.load_pics import load_pics


# Loading the dataset
comp = "iKosh"
train_file = "train_signs.h5"
test_file = "test_signs.h5"
X_train, Y_train, X_test, Y_test, classes = load_pics(comp, train_file, test_file)

# Example of a picture
index = 0
plt.imshow(X_train[index])
print ("y = " + str(np.squeeze(X_train[:, index])))

# Flatten the training and test images
X_train_flatten = X_train.reshape(X_train.shape[0], -1).T
X_test_flatten = X_test.reshape(X_test.shape[0], -1).T

# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train, 6)
Y_test = convert_to_one_hot(Y_test, 6)

print("number of training examples = " + str(X_train.shape[1]))
print("number of test examples = " + str(X_test.shape[1]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

### TENSORFLOW MODEL ###
print('\n' + "\033[1m" + " TENSORFLOW MODEL:" + "\033[0m")
parameters = model(X_train, Y_train, X_test, Y_test)