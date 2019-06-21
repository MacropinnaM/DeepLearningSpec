import matplotlib.pyplot as plt
import numpy as np

from AA_LogReg._m_logreg import logreg
from AA_LogReg.load_dataset import load_dataset

# Loading the data (cat/non-cat)
data_folder = '/Users/lilkosh/PycharmProjects/DeepLearningSpec/_Data/'
train_file = 'train_catvnoncat.h5'
test_file = 'test_catvnoncat.h5'
train_x, train_y, test_x, test_y, classes = load_dataset(data_folder, train_file, test_file)

# Example of a picture
index = 25
plt.imshow(train_x[index])
plt.show()
print("y = " + str(train_y[:, index]) + ", it's a '" +
      classes[np.squeeze(train_y[:, index])].decode("utf-8") + "' picture." + '\n')

m_train = train_x.shape[0]
m_test = test_x.shape[0]
num_px = train_x.shape[1]

print('\n' + "\033[1m" + "Initial shapes: " + "\033[0m")
print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x shape: " + str(train_x.shape))
print("train_set_y shape: " + str(train_y.shape))
print("test_set_x shape: " + str(test_x.shape))
print("test_set_y shape: " + str(test_y.shape))

# Reshape the training and test examples
train_x_flatten = train_x.reshape(train_x.shape[0], -1).T
test_x_flatten = test_x.reshape(test_x.shape[0], -1).T

print('\n' + "\033[1m" + "Shapes after processing: " + "\033[0m")
print("train_set_x_flatten shape: " + str(train_x_flatten.shape))
print("train_set_y shape: " + str(train_y.shape))
print("test_set_x_flatten shape: " + str(test_x_flatten.shape))
print("test_set_y shape: " + str(test_y.shape))
print("sanity check after reshaping: " + str(train_x_flatten[0:5, 0]))

# Standardize the datasets
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

# Train the model
print('\n' + "\033[1m" + "Train the model: " + "\033[0m")
d = logreg(train_x, train_y, test_x, test_y,
           num_iterations=2000, lr=0.005, print_cost=True)

# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

# Further analysis -- Learning rate choice
print('\n' + "\033[1m" + "Choosing the learning rate: " + "\033[0m")
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("learning rate is: " + str(i))
    models[str(i)] = logreg(train_x, train_y, test_x, test_y,
                            num_iterations=1500, lr=i, print_cost=False)
    print("------------------------------------")

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()