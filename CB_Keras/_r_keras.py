# Loading the data (signs)
from CB_Keras.load_pics import load_pics

comp = "iKosh"
train_file = "train_happy.h5"
test_file = "test_happy.h5"
X_train, Y_train, X_test, Y_test, classes = load_pics(comp, train_file, test_file)

# Normalize image vectors and reshape label vectors
X_train = X_train/255.
X_test = X_test/255.
Y_train = Y_train.T
Y_test = Y_test.T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

