from keras import Input, Model
from keras.initializers import glorot_uniform
from keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Flatten, \
    Dense

from CC_ResNet.convolutional_block import convolutional_block
from CC_ResNet.identity_block import identity_block


def ResNet50(input_shape=(64, 64, 3), classes=6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    f = 3
    stage, filters = 2, [64, 64, 256]
    X = convolutional_block(X, f, filters, stage, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    stage, filters = 3, [128, 128, 512]
    X = convolutional_block(X, f, filters, stage, block='a', s=2)
    X = identity_block(X, f, filters, stage, block='b')
    X = identity_block(X, f, filters, stage, block='c')
    X = identity_block(X, f, filters, stage, block='d')

    stage, filters = 4, [256, 256, 1024]
    X = convolutional_block(X, f, filters, stage, block='a', s=2)
    X = identity_block(X, f, filters, stage, block='b')
    X = identity_block(X, f, filters, stage, block='c')
    X = identity_block(X, f, filters, stage, block='d')
    X = identity_block(X, f, filters, stage, block='e')
    X = identity_block(X, f, filters, stage, block='f')

    stage, filters = 5, [512, 512, 2048]
    X = convolutional_block(X, f, filters, stage, block='a', s=2)
    X = identity_block(X, f, filters, stage, block='b')
    X = identity_block(X, f, filters, stage, block='c')

    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model