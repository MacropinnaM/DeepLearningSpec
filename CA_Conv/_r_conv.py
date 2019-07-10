import numpy as np
import matplotlib.pyplot as plt

from CA_Conv._m_conv import model
from CA_Conv.convert_to_one_hot import convert_to_one_hot
from CA_Conv.load_pics import load_pics


# Loading the data (signs)
comp = "iKosh"
train_file = "train_signs.h5"
test_file = "test_signs.h5"
X_train, Y_train, X_test, Y_test, classes = load_pics(comp, train_file, test_file)

# Example of a picture
index = 10
plt.imshow(X_train[index])
print("y = " + str(np.squeeze(Y_train[:, index])))
plt.show()

# Examine the shapes of th data
X_train = X_train/255.
X_test = X_test/255.
Y_train = convert_to_one_hot(Y_train, 6).T
Y_test = convert_to_one_hot(Y_test, 6).T
print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))
conv_layers = {}

# TRAIN THE MODEL
print('\n' + "\033[1m" + " TRAIN THE MODEL:" + "\033[0m")
_, _, parameters = model(X_train, Y_train, X_test, Y_test)