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
from inference import infer
from util import batch_generator 
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
    """Load data into memory and cache it."""
    # Load all images into memory.
    paths = glob.glob(os.path.join(root, "**/*.jpg"), recursive=True)
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        data = p.map(read_img, paths)
        images = list(zip(paths, data))
        
    # Split test and training data according to Kaggle's rules.
    test_directory = 'test_stg1'
    train = [(path, img) for path, img in images if test_directory not in path]
    test = [(path, img) for path, img in images if test_directory in path]

    # Create class label for every training image in a horrible way for the lulz.
    labels = [[CLASS_NAMES.index(class_name) for class_name in CLASS_NAMES if class_name in path] for path in [_[0] for _ in train]]
   
    # One-hot encode labels. # TODO: use sparse with tensorflow instead
    y_train = OneHotEncoder(sparse=False).fit_transform(np.array(labels))
    
    # Stack images as ndarrays.
    X_train = np.array([img for _, img in train])
    X_test = np.array([img for _, img in test])
  
    # Calculate channel-wise mean for training data.
    rgb_mean = X_train.mean(axis=(0, 1, 2))
    
    # Color-normalize images.
    X_train -= rgb_mean
    X_test -= rgb_mean

    return X_train, y_train, X_test

def new_run(X_train, y_train, X_val, y_val, model_savename):
    """Trains and saves a model with given training data."""
    tf.reset_default_graph()
    batches = batch_generator((X_train, y_train), batch_size=128)
    
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
        
        print("Starting training...")
        for _ in range(int(1e7)):
            X_batch, y_batch = next(batches)
            _, summary, i = sess.run([train_step, merged_summary, global_step], feed_dict={X: X_batch, target: y_batch, model.keep_prob: 0.5})
            summary_writer.add_summary(summary, i)
            if i > 100000 and i % 1000 == 0:
                probs_val = infer(sess, model, X_val)
                # TODO: compare with y_val to see if we should stop early


        # TODO run accuracy on whole validation set
        probs_val = infer(sess, model, X_val)
        # TODO: define tf ops for this total accuracy

        saver.save(sess, model_savename)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    aarg = argparser.add_argument
    aarg("--dataset-dir", required=True, help="dataset directory")
    aarg("--run-name", help="name for model checkpoints")
    args = argparser.parse_args()

    X_train, y_train, X_test = load_data(args.dataset_dir)
   
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    model_savename = args.run_name if args.run_name else timestamp
    print("Model base savename: ", model_savename)
    
    # Train n_splits models with X_train split into training and validation set
    # in n_splits different ways.
    kfold = KFold(n_splits=3)
    for run_index, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
        # Start a new run with this split. 
        new_run(X_train[train_idx], 
                y_train[train_idx],
                X_train[val_idx],
                y_train[val_idx],
                "{}_val_{}".format(model_savename, run_index))
