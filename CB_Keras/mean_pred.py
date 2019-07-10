import keras.backend as K


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)