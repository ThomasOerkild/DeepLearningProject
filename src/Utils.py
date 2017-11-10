import numpy as np
from sklearn.preprocessing import OneHotEncoder


def convert_label_to_one_hot(image, num_classes=12):
    n, m = image.shape
    return np.array(OneHotEncoder(n_values=num_classes, sparse=False).fit_transform(image.reshape(-1, 1))).reshape(n, m, num_classes)