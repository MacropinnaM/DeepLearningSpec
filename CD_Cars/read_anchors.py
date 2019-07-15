import numpy as np


def read_anchors(anchors_path):
    """

    :param anchors_path:
    :return:
    """

    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)

    return anchors