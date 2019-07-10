# Loading the data (signs)
from CC_ResNet.load_pics import load_pics

comp = "iKosh"
train_file = "train_signs.h5"
test_file = "test_signs.h5"
X_train, Y_train, X_test, Y_test, classes = load_pics(comp, train_file, test_file)