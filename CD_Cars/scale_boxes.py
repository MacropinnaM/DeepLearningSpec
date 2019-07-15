from keras import backend as K


def scale_boxes(boxes, image_shape):
    """
    Scales the predicted boxes in order to be drawable on the image
    :param boxes: 
    :param image_shape:
    :return:
    """

    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims

    return boxes