import imghdr
import numpy as np
from PIL import Image


def preprocess_image(img_path, model_image_size):
    """

    :param img_path:
    :param model_image_size:
    :return:
    """

    image_type = imghdr.what(img_path)
    image = Image.open(img_path)
    resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
    image_data = np.array(resized_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    return image, image_data