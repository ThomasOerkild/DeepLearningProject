import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
from skimage.io import imread
import glob
import tensorflow as tf


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
        

def unpool_with_argmax(pool, ind, name = None, ksize=[1, 2, 2, 1]):

    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(name):
        input_shape = pool.get_shape().as_list()
        print("input_shape: {0}, ind: {1}".format(pool.shape, ind.shape))
        output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])

        flat_input_size = tf.reduce_prod(input_shape)
        flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

        pool_ = tf.reshape(pool, [flat_input_size])
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=ind.dtype), shape=[input_shape[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [flat_input_size, 1])
        ind_ = tf.reshape(ind, [flat_input_size, 1])
        ind_ = tf.concat([b, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=flat_output_shape)
        ret = tf.reshape(ret, output_shape)
        return ret