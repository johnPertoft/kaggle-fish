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
NUM_CLASSES = len(CLASS_NAMES)

def read_img(path):
    img = imread(path)
    img = resize(img, IMG_SHAPE)
    return img.astype("float32") / 255

def preprocess(x, rgb_mean):
    return x - rgb_mean

def load_train_data(training_rootdir):
    X_train, y_train = [], []
    for i, class_dir in enumerate(CLASS_NAMES):
        img_paths = glob.glob(os.path.join(training_rootdir, class_dir, "*.jpg"))
        for img_path in img_paths:
            X_train.append(read_img(img_path))
            y_train.append(i)

    X_train = np.array(X_train)
    y_train = OneHotEncoder(sparse=False).fit_transform(np.array(y_train)[:, np.newaxis])

    rgb_mean = X_train.mean(axis=(0, 1, 2)) # TODO: Maybe remove per image instead?
    X_train = preprocess(X_train, rgb_mean)

    return X_train, y_train, rgb_mean

def load_test_data(test_dir, rgb_mean):
    X_test = [read_img(p) for p in glob.glob(os.path.join(test_dir, "*.jpg"))]
    return preprocess(np.array(X_test), rgb_mean)

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
    aarg("--run-name", help="Name for model checkpoints")
    #aarg("--restart-training"
    args = argparser.parse_args()

    print("Loading train and test data.")
    X_train, y_train, rgb_mean = load_train_data(os.path.join(args.dataset_dir, "train"))
    X_test = load_test_data(os.path.join(args.dataset_dir, "test_stg1"), rgb_mean)
    print("Loaded data.")
  
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        # Create the model
        X = tf.placeholder(tf.float32, (None, IMG_SHAPE[0], IMG_SHAPE[1], 3))
        target = tf.placeholder(tf.float32, (None, NUM_CLASSES))
        model = Fishmodel(X, num_classes=NUM_CLASSES)
        
        # Cross entropy loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(model.logits, 
                                                                target, 
                                                                name="cross_entropy")
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
