import argparse
import datetime
import glob
import os
import signal
import sys
import multiprocessing

import numpy as np
import tensorflow as tf
import cv2
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from joblib import Memory

from model import Fishmodel

memory = Memory('cache', verbose=0)
SUMMARY_DIR = "/tmp/kaggle-fish"
CHECKPOINT_DIR = "model-checkpoints"
IMG_SHAPE = (64, 64)  # TODO: too small?
CLASS_NAMES = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]
NUM_CLASSES = len(CLASS_NAMES)


def read_img(path, shape=IMG_SHAPE):
    img = cv2.imread(path)
    img = cv2.resize(img, shape)
    return img.astype(np.float32) / 255


@memory.cache
def load_data(root):

    # Load all images into memory.
    paths = glob.glob(os.path.join(root, "**/*.jpg"), recursive=True)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        data = p.map(read_img, paths)
        images = list(zip(paths, data))
        
    # Split test and training data according to Kaggle's rules.
    test_directory = 'test_stg1'
    train = [_ for _ in images if test_directory not in _[0]]
    test = [_ for _ in images if test_directory in _[0]]

    # Create class label for every training image in a horrible way for the lulz.
    labels = [[CLASS_NAMES.index(class_name) for class_name in CLASS_NAMES if class_name in path] for path in [_[0] for _ in train]]
   
    # One-hot encode labels.
    y_train = OneHotEncoder(sparse=False).fit_transform(np.array(labels))
    
    # Stack images as ndarrays.
    X_train = np.array([_[1] for _ in train])
    X_test = np.array([_[1] for _ in test])
  
    # Calculate channel-wise mean for training data.
    rgb_mean = X_train.mean(axis=(0, 1, 2))  # TODO: Maybe remove per image instead?
    
    # Color-normalize images.
    X_train -= rgb_mean
    X_test -= rgb_mean

    return X_train, y_train, X_test


def batch_generator(X_train, y_train, batch_size):
    N = X_train.shape[0]
    while True:
        X_train, y_train = shuffle(X_train, y_train)
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            yield X_train[i:j], y_train[i:j]


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    aarg = argparser.add_argument
    aarg("--dataset-dir", required=True, help="dataset directory")
    aarg("--run-name", help="name for model checkpoints")
    args = argparser.parse_args()

    X_train, y_train, X_test = load_data(args.dataset_dir)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        # Create the model
        X = tf.placeholder(tf.float32, (None, IMG_SHAPE[0], IMG_SHAPE[1], 3))
        target = tf.placeholder(tf.float32, (None, NUM_CLASSES))
        model = Fishmodel(X, num_classes=NUM_CLASSES)

        # Cross entropy loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(model.logits, target, name="cross_entropy")
        loss = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")
        tf.scalar_summary("mean cross entropy loss", loss)

        # Summary reports for tensorboard
        merged_summary = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(SUMMARY_DIR, sess.graph)

        saver = tf.train.Saver(tf.all_variables())
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        model_savename = args.run_name if args.run_name else timestamp
        print("Model savename: ", model_savename)

        # Save model on ctrl-c
        def exit_handler(signal, frame):
            print("Saving model checkpoint")
            saver.save(sess, os.path.join(CHECKPOINT_DIR, model_savename))
            sys.exit()

        signal.signal(signal.SIGINT, exit_handler)

        print("Starting training.")
        print("Exit with CTRL-C, model will be saved on exit.")

        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
        batches = batch_generator(X_train, y_train, batch_size=128)
        sess.run(tf.initialize_all_variables())
        while True:
            X_batch, y_batch = next(batches)
            feed_dict = {X: X_batch, target: y_batch, model.keep_prob: 0.5}
            _, loss_val = sess.run([train_step, loss], feed_dict=feed_dict)
            print(loss_val)
