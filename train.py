import signal
import sys
import os
import glob
import argparse
import datetime

from skimage.io import imread
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder
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
    y_train = OneHotEncoder().fit_transform(np.array(y_train)[:, np.newaxis])

    rgb_mean = x_train.mean(axis=(0, 1, 2)) # TODO: Maybe remove per image instead?
    x_train = preprocess(x_train, rgb_mean)

    return x_train, y_train, rgb_mean

def load_test_data(test_dir, rgb_mean):
    x_test = [read_img(p) for p in glob.glob(os.path.join(test_dir, "*.jpg"))]
    return preprocess(np.array(x_test), rgb_mean)

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
        X = tf.placeholder(tf.float32, (None, IMG_SHAPE[0], IMG_SHAPE[1]))
        fish_model = Fishmodel(X)
        
        # Summary reports for tensorboard
        merged_summary = tf.merge_all_summaries()
        summary_writer = tf.train.SummaryWriter(SUMMARY_DIR, sess.graph)

        saver = tf.train.Saver(tf.all_variables())
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        model_savename = args.run_name if args.run_name else timestamp
        
        # Save model on ctrl-c
        def exit_handler(signal, frame):
            saver.save(sess, os.path.join(CHECKPOINT_DIR, model_savename)) 
            sys.exit()
        signal.signal(signal.SIGINT, exit_handler)
        
        print("Starting training.")
        print("Exit with CTRL-C, model will be saved on exit.")

        while True:
            pass
