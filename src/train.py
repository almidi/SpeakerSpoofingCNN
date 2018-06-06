import os
import sys
import numpy as np
import tensorflow as tf

from src.model import CNN
from lib.model_io import get_modle_id

model_id = get_modle_id()

#Enable gpu memory growth
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.log_device_placement=True
# Create the network
# network = CNN(cfg, model_id) TODO What is cfg ??
network = CNN(model_id)

# Define the train computation graph
network.define_train_operations()

# Train the network
sess = tf.Session(config=config)
try:
    # network.train(cfg, coord, sess) TODO What is cfg , coord ?
    network.train(sess)
except KeyboardInterrupt:
    print()
finally:
    sess.close() 










