import numpy as np
import tensorflow as tf


def conv2d(prev, name, shape, stride=(1, 1, 1, 1)):
    with tf.variable_scope(name) as scope:
        kernel = tf.Variable(
            tf.truncated_normal(
                shape, dtype=tf.float32, stddev=1e-1),
            name="weights")

        conv = tf.nn.conv2d(prev, kernel, stride, padding="SAME")

        biases = tf.Variable(
            tf.constant(
                0.0, shape=[shape[-1]], dtype=tf.float32),
            name="biases")

        bias_add = tf.nn.bias_add(conv, biases)

        return tf.nn.relu(bias_add, name=name)


def max_pool2d(prev, name, shape, stride=[1, 2, 2, 1]):
    return tf.nn.max_pool(prev, shape, stride, padding="SAME", name=name)


def dense(prev, name, output_dim, activation=True):
    with tf.variable_scope(name) as scope:
        flatten_first = len(prev.get_shape().as_list()) > 2
        if flatten_first:
            shape = np.prod(prev.get_shape().as_list()[1:])
            prev = tf.reshape(prev, [-1, shape])
        else:
            shape = prev.get_shape().as_list()[1]

        weights = tf.Variable(
            tf.truncated_normal(
                (shape, output_dim), dtype=tf.float32, stddev=1e-1),
            name="weights")

        biases = tf.Variable(
            tf.constant(
                1.0, shape=[output_dim], dtype=tf.float32),
            name="biases")

        bias_add = tf.nn.bias_add(tf.matmul(prev, weights), biases)

        return tf.nn.relu(bias_add, name=name) if activation else bias_add


class Fishmodel:
    def __init__(self, X, num_classes):
        # TODO: just some random model to start with atm, probably need something more complex for this task

        self.X = X
        self.keep_prob = tf.placeholder(tf.float32)

        self.conv1 = conv2d(self.X, "conv1", shape=(3, 3, 3, 16))
        self.conv2 = conv2d(self.conv1, "conv2", shape=(3, 3, 16, 32))
        self.mpool1 = max_pool2d(
            self.conv2, "max_pool1", shape=(1, 2, 2, 1), stride=(1, 2, 2, 1))

        self.conv3 = conv2d(self.mpool1, "conv3", shape=(3, 3, 32, 64))
        self.conv4 = conv2d(self.conv3, "conv4", shape=(3, 3, 64, 64))
        self.mpool2 = max_pool2d(
            self.conv4, "max_pool2", shape=(1, 2, 2, 1), stride=(1, 2, 2, 1))
        self.mpool2_drop = tf.nn.dropout(self.mpool2, self.keep_prob)

        self.dense1 = dense(self.mpool2_drop, "dense1", 2048)
        self.dense2 = dense(self.dense1, "dense2", 2048)
        self.logits = dense(
            self.dense2, "logits", num_classes, activation=False)
        self.softmax = tf.nn.softmax(self.logits, name="softmax")
