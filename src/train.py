import os
import sys
import numpy as np
import tensorflow as tf

from src.model import CNN
from lib.model_io import *
from src.predict import predict


def train(epochs=12, model='baseline', early_stop=4, learning_rate=0.0001, batch_norm=True, batch_size=256):
    model_id = get_model_id()

    # Enable gpu memory growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement=True

    # Create the network
    # network = CNN(cfg, model_id) TODO What is cfg ??
    network = CNN(model_id, model=model, epochs=epochs, early_stop=early_stop, learning_rate=learning_rate,
                  batch_norm=batch_norm, batch_size=batch_size)

    # Define the train computation graph
    network.define_train_operations()

    # Train the network
    sess = tf.Session(config=config)

    try:
        losses = network.train(sess)

        dict = {"model_id": model_id, "model": model, "train_losses": [r[0] for r in losses],
                "valid_losses": [r[1] for r in losses], "epochs": epochs, "early_stop": early_stop,
                "learning_rate": learning_rate, "batch_norm": batch_norm}
    except KeyboardInterrupt:
        print()
    finally:
        tf.reset_default_graph
        network.reset()
        sess.close()

    sess.close()
    return dict

if __name__ == "__main__":
    dict = train(epochs = 64,model='baseline',early_stop=12,learning_rate=0.0001,batch_norm=True,batch_size=256)
    save_model_stats(dict.get('model_id'), dict)
    dict['accuracy'] = predict(epochs = 64,model='baseline',early_stop=12,learning_rate=0.0001,batch_norm=True,batch_size=256)
    # Save data
    save_model_stats(dict.get('model_id'), dict)
