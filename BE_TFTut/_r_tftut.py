import tensorflow as tf


from BE_TFTut.load_pics import load_pics


# Loading the data
comp = 'iKosh'
train_file, test_file = 'train_signs.h5', 'test_signs.h5'
train_x, train_y, test_x, test_y, classes = load_pics(comp, train_file, test_file)