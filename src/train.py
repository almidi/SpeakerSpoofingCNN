import os
import numpy as np
import tensorflow as tf

sys.path.insert(0, './lib')

from model import CNN
from model_io import get_modle_id

model_id = get_modle_id()

# Create the network
# network = CNN(cfg, model_id) TODO What is cfg ??
network = CNN(model_id)

# Define the train computation graph
network.define_train_operations()

# Train the network
sess = tf.Session()
try:
    # network.train(cfg, coord, sess) TODO What is cfg , coord ?
    network.train(sess)
except KeyboardInterrupt:
    print()
finally:
    sess.close() 










