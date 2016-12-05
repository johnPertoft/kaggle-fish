import numpy as np
import tensorflow as tf


def conv2d(previous, shape, stride=(1, 1, 1, 1), activation=tf.nn.relu):
    kernel = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    conv = tf.nn.conv2d(previous, kernel, stride, padding="SAME")
    biases = tf.Variable(tf.constant(0.0, shape=[shape[-1]]))
    bias_add = tf.nn.bias_add(conv, biases)
    return activation(bias_add) if activation else bias_add


def max_pool2d(previous, shape, stride):
    return tf.nn.max_pool(previous, shape, stride, padding="SAME")


def dense(previous, output_dim, activation=tf.nn.relu):
    # TODO Document why flattening is ever needed.
    flatten_first = len(previous.get_shape().as_list()) > 2
    if flatten_first:
        shape = np.prod(previous.get_shape().as_list()[1:])
        previous = tf.reshape(previous, [-1, shape])
    else:
        shape = previous.get_shape().as_list()[1]

    weights = tf.Variable(tf.truncated_normal((shape, output_dim), stddev=0.1))
    biases = tf.Variable(tf.constant(1.0, shape=[output_dim]))
    bias_add = tf.nn.bias_add(tf.matmul(previous, weights), biases)
    return activation(bias_add) if activation else bias_add


class Fishmodel:
    def __init__(self, X, num_classes):
        self.conv1 = conv2d(X, (3, 3, 3, 16))
        self.conv2 = conv2d(self.conv1, (3, 3, 16, 32))
        self.mpool1 = max_pool2d(self.conv2, (1, 2, 2, 1), (1, 2, 2, 1))

        self.conv3 = conv2d(self.mpool1, (3, 3, 32, 64))
        self.conv4 = conv2d(self.conv3, (3, 3, 64, 64))
        self.mpool2 = max_pool2d(self.conv4, (1, 2, 2, 1), (1, 2, 2, 1))

        self.keep_prob = tf.placeholder(tf.float32)
        self.dropout = tf.nn.dropout(self.mpool2, self.keep_prob)

        self.dense1 = dense(self.dropout, 2048)
        self.dense2 = dense(self.dense1, 2048)
        self.logits = dense(self.dense2, num_classes, activation=None)
        self.softmax = tf.nn.softmax(self.logits)
