import matplotlib.pyplot as plt

from AC_DeepNet._m_deepnet import l_layer_model
from AC_DeepNet.load_cats import load_cats
from AC_DeepNet.predict import predict
from AC_DeepNet.two_layer_model import two_layer_model

# Loading the data (cat/non-cat)
data_folder = '/Users/lilkosh/PycharmProjects/DeepLearningSpec/_Data/'
train_file, test_file = 'train_catvnoncat.h5', 'test_catvnoncat.h5'
train_x, train_y, test_x, test_y, classes = load_cats(data_folder, train_file, test_file)

# Example of a picture
index = 5
plt.imshow(train_x[index])
print ("y = " + str(train_y[0, index]) + ". It's a " + classes[train_y[0, index]].decode("utf-8") + " picture.")
print(train_x[index].shape)

# Explore your dataset
m_train = train_x.shape[0]
num_px = train_x.shape[1]
m_test = test_x.shape[0]

print('\n' + "\033[1m" + "Shapes before processing: " + "\033[0m")
print("Number of training examples: " + str(m_train))
print("Number of testing examples: " + str(m_test))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_x_orig shape: " + str(train_x.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x_orig shape: " + str(test_x.shape))
print("test_y shape: " + str(test_y.shape))
print()

# Reshape the training and test examples
train_x_flatten = train_x.reshape(train_x.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x.reshape(test_x.shape[0], -1).T

# Standardize data to have feature values between 0 and 1.
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

print('\n' + "\033[1m" + "Shapes after processing: " + "\033[0m")
print("train_x's shape: " + str(train_x.shape))
print("test_x's shape: " + str(test_x.shape))
print()

### TWO-LAYER NEURAL NETWORK ###
print('\n' + "\033[1m" + " TWO-LAYER NEURAL NETWORK: " + "\033[0m")
n_x = 12288  # num_px * num_px * 3
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)


parameters = two_layer_model(train_x, train_y, layers_dims=(n_x, n_h, n_y),
                             num_iterations=2000, print_cost=True)

predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)
print()

### L-LAYER NEURAL NETWORK ###
print('\n' + "\033[1m" + " L-LAYER NEURAL NETWORK: " + "\033[0m")
layers_dims = [12288, 20, 7, 5, 1]  # 4-layer model

parameters = l_layer_model(train_x, train_y, layers_dims,
                           num_iterations=2000, print_cost=True)
predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)