def read_classes(classes_path):
    """

    :param classes_path:
    :return:
    """

    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    
    return class_names