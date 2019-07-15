# Loading the data (signs)
from CC_ResNet._m_resnet import ResNet50
from CC_ResNet.convert_to_one_hot import convert_to_one_hot
from CC_ResNet.load_pics import load_pics

comp = "iKosh"
train_file = "train_signs.h5"
test_file = "test_signs.h5"
X_train, Y_train, X_test, Y_test, classes = load_pics(comp, train_file, test_file)

# Normalize image vectors
X_train = X_train/255.
X_test = X_test/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train, 6).T
Y_test = convert_to_one_hot(Y_test, 6).T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

# TRAIN 50 LAYERS RESNET
print('\n' + "\033[1m" + " TRAIN 50 LAYERS RESNET:" + "\033[0m")
model = ResNet50(input_shape=(64, 64, 3), classes=6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=4, batch_size=32)
preds = model.evaluate(X_test, Y_test)
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
