import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
from skimage.io import imread
import glob


def convert_annotation_to_one_hot(image, num_classes=12):
    n, m = image.shape
    return np.array(OneHotEncoder(n_values=num_classes, sparse=False).fit_transform(image.reshape(-1, 1))).reshape(n, m, num_classes)

def load_images(directory):
    image_paths = glob.glob(os.path.join(directory, "*.png"))
    images = []
    for image_path in image_paths:
        images.append(imread(image_path)[0:352,:])
    return images

def load_annotations(directory):
    annotations = load_images(directory)
    return list(map(convert_annotation_to_one_hot, annotations))

class BatchProcessor:

    def __init__(self):
        self.current_index = 0


    def get_next_batch(self,X_train, y_train, batch_size):
        old_index = self.current_index
        self.current_index += batch_size

        if old_index % len(X_train) < self.current_index % len(X_train):
            return (
                X_train[old_index % len(X_train):self.current_index % len(X_train)],
                y_train[old_index % len(X_train):self.current_index % len(X_train)])
        else:
            index_to_end = range((old_index % len(X_train)), len(X_train))
            index_from_start = range(0, self.current_index % len(X_train))
            return (np.concatenate([X_train[index_to_end], X_train[index_from_start]]),
                    np.concatenate([y_train[index_to_end], y_train[index_from_start]]))