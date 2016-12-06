import argparse
import datetime
import glob
import os
import signal
import sys
import multiprocessing
import shutil

import numpy as np
import tensorflow as tf
import cv2
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
from joblib import Memory

from config import CLASS_NAMES, NUM_CLASSES
from model import Fishmodel

cache = Memory('cache', verbose=0)

IMAGE_SHAPE = (64, 64)
CHECKPOINT_DIR = 'models'
SUMMARY_DIR = 'tensorboard'
if os.path.isdir(SUMMARY_DIR):
    shutil.rmtree(SUMMARY_DIR)
else:
    os.mkdir(SUMMARY_DIR)
    

if not os.path.isdir(CHECKPOINT_DIR):
    os.mkdir(CHECKPOINT_DIR)
 

def read_img(path, shape=IMAGE_SHAPE):
    img = cv2.imread(path)
    img = cv2.resize(img, shape)
    return img.astype(np.float32) / 255


@cache.cache
def load_data(root, shape=IMAGE_SHAPE):

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

def new_run(X_train, y_train, model_savename):
    tf.reset_default_graph()
    batches = batch_generator(X_train, y_train, batch_size=128)
    
    with tf.Session() as sess: 
        # Create the model
        X = tf.placeholder(tf.float32, (None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))
        target = tf.placeholder(tf.float32, (None, NUM_CLASSES))
        model = Fishmodel(X, num_classes=NUM_CLASSES)

        saver = tf.train.Saver(tf.global_variables())
        
        # Cross entropy loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(model.logits, target, name="cross_entropy")
        loss = tf.reduce_mean(cross_entropy, name="cross_entropy_mean")

        # Accuracy
        corrects = tf.equal(tf.argmax(model.softmax, 1), tf.argmax(target, 1))
        accuracy = tf.reduce_mean(tf.cast(corrects, tf.uint8))

        # Summary reports for tensorboard
        tf.scalar_summary("Mean Cross Entropy Loss", loss)
        tf.scalar_summary("Accuracy", accuracy)
        merged_summary = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(SUMMARY_DIR, sess.graph)
        
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step)
        sess.run(tf.global_variables_initializer())
        
        print("Starting trainigni")
        for val_i in range(int(1e7)):
            X_batch, y_batch = next(batches)
            _, summary, i = sess.run([train_step, merged_summary, global_step], feed_dict={X: X_batch, target: y_batch, model.keep_prob: 0.5})
            summary_writer.add_summary(summary, i)
   
        # TODO run accuracy on whole validation set

        saver.save(sess, model_savename + str(val_i))



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
   
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model_savename = args.run_name if args.run_name else timestamp
    print("Model savename: ", model_savename)

    with tf.Session() as sess:
        kfold = KFold(n_splits=3)
        for train_idx, val_idx in kfold.split(X_train):
            # Start a new run with this split. 
            new_run(X_train[train_idx], y_train[train_idx],  model_savename)
