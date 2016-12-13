import numpy as np
import tensorflow as tf


def conv2d(previous, shape, name, stride=(1, 1, 1, 1), activation=tf.nn.relu):
    with tf.variable_scope(name):
        kernel = tf.get_variable('Kernel', 
                                 shape, 
                                 initializer=tf.contrib.layers.xavier_initializer()),
        convolved = tf.nn.bias_add(
            tf.nn.conv2d(previous, kernel, stride, padding="SAME"), 
            tf.get_variable('Biases', shape[-1], initializer=tf.constant_initializer(0.0)))
        return activation(convolved) if activation else convolved


def max_pool(previous, shape, name):
    with tf.variable_scope(name):
        return tf.nn.max_pool(previous, shape, shape, padding="SAME")


def flatten(previous, name):
    with tf.variable_scope(name):
        shape = np.prod(previous.get_shape().as_list()[1:])  # Skip batch dimension
        return tf.reshape(previous, (-1, shape))


def dense(name, previous, output_dim, activation=tf.nn.relu):
    with tf.variable_scope(name):
        input_dim = previous.get_shape().as_list()[1]
        product = tf.nn.bias_add(
            tf.matmul(previous, 
                      tf.get_variable('Weights', 
                                      (input_dim, output_dim), 
                                      initializer=tf.contrib.layers.xavier_initializer())),
            tf.get_variable('Biases', output_dim, initializer=tf.constant_initializer(0.0)))
        return activation(product) if activation else product


class Fishmodel:
    @classmethod
    def from_file(cls, path):
        pass # TODO 

    def __init__(self, X, num_classes):
        self.conv1 = conv2d(X, (3, 3, 3, 16), "Convolution1")
        self.conv2 = conv2d(self.conv1, (3, 3, 16, 32), "Convolution2")
        self.pool1 = max_pool(self.conv2, (1, 2, 2, 1), "MaxPooling1")

        self.conv3 = conv2d(self.pool1, (3, 3, 32, 64), "Convolution3")
        self.conv4 = conv2d(self.conv3, (3, 3, 64, 64), "Convolution4")
        self.pool2 = max_pool(self.conv4, (1, 2, 2, 1), "MaxPooling2")

        self.keep_prob = tf.placeholder(tf.float32)
        self.dropout = tf.nn.dropout(self.pool2, self.keep_prob)
        
        self.dense1 = dense(flatten(self.dropout), 2048, "Dense1")
        self.dense2 = dense(self.dense1, 2048, "Dense2")
        self.logits = dense(self.dense2, num_classes, "Dense3", activation=None)
        self.softmax = tf.nn.softmax(self.logits)
