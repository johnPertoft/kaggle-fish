import signal
import sys
import os

import tensorflow as tf

from model import Fishmodel

if __name__ == "__main__":
    #X = tf.placeholder(tf.float32, (None,
    #fish_model = Fishmodel(X)
   
    saver = tf.train.Saver(tf.all_variables())

    def exit_handler(signal, frame):
        # TODO: save checkpoint
        sys.exit()
    
    signal.signal(signal.SIGINT, exit_handler)

    while True:
        pass
