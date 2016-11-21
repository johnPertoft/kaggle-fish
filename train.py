import signal
import sys
import os
import glob
import argparse
import datetime

from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np

from model import Fishmodel

SUMMARY_DIR = "/tmp/kaggle-fish"
CHECKPOINT_DIR = "model-checkpoints"
IMG_SHAPE = (64, 64) # TODO: too small?
CLASS_NAMES = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]

def read_img(path):
    img = imread(path)
    img = resize(img, IMG_SHAPE)
    return img.astype("float32") / 255

def preprocess(x, rgb_mean):
    return x - rgb_mean

def load_train_data(training_rootdir):
    x_train, y_train = [], []
    for i, class_dir in enumerate(CLASS_NAMES):
        img_paths = glob.glob(os.path.join(training_rootdir, class_dir, "*.jpg"))
        for img_path in img_paths:
            x_train.append(read_img(img_path))
            y_train.append(i)

    x_train = np.array(x_train)
    y_train = OneHotEncoder(sparse=False).fit_transform(np.array(y_train)[:, np.newaxis])

    rgb_mean = x_train.mean(axis=(0, 1, 2)) # TODO: Maybe remove per image instead?
    x_train = preprocess(x_train, rgb_mean)

    return x_train, y_train, rgb_mean

def load_test_data(test_dir, rgb_mean):
    x_test = [read_img(p) for p in glob.glob(os.path.join(test_dir, "*.jpg"))]
    return preprocess(np.array(x_test), rgb_mean)

def batch_generator(x_train, y_train, batch_size):
    N = x_train.shape[0]
    while True:
        x_train, y_train = shuffle(x_train, y_train)
        for i in range(0, N, batch_size):
            j = min(i + batch_size, N)
            yield x_train[i:j], y_train[i:j]

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    aarg = argparser.add_argument
    aarg("--dataset-dir", required=True, help="dataset directory")
    aarg("--run-name", help="Name for model checkpoints")
    #aarg("--restart-training"
    args = argparser.parse_args()

    print("Loading train and test data.")
    x_train, y_train, rgb_mean = load_train_data(os.path.join(args.dataset_dir, "train"))
    x_test = load_test_data(os.path.join(args.dataset_dir, "test_stg1"), rgb_mean)
    print("Loaded data.")
  
    with tf.Session() as sess:
        # Create the model
        X = tf.placeholder(tf.float32, (None, IMG_SHAPE[0], IMG_SHAPE[1], 3))
        target = tf.placeholder(tf.float32, (None, 8))
        fish_model = Fishmodel(X, num_classes=8)
        
        # Cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits(fish_model.logits, target)

        # Summary reports for tensorboard
        merged_summary = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(SUMMARY_DIR, sess.graph)

        saver = tf.train.Saver(tf.all_variables())
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        model_savename = args.run_name if args.run_name else timestamp
        print("Model savename: ", model_savename)
        
        # Save model on ctrl-c
        def exit_handler(signal, frame):
            saver.save(sess, os.path.join(CHECKPOINT_DIR, model_savename)) 
            sys.exit()
        signal.signal(signal.SIGINT, exit_handler)
        
        print("Starting training.")
        print("Exit with CTRL-C, model will be saved on exit.")

        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
        batches = batch_generator(x_train, y_train, batch_size=128)
        sess.run(tf.initialize_all_variables())
        while True:
            x_batch, y_batch = next(batches)
            train_step.run(feed_dict={X: x_batch, target: y_batch})
            # TODO: write accuracy to summary write sometimes
