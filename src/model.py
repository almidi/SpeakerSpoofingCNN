from __future__ import division, print_function

import sys
import time
from datetime import datetime
import math
import tensorflow as tf
from lib.model_io import *
from src.fdata import fdata
import progressbar
import random
from lib.precision import _FLOATX


class CNN(object):

    def __init__(self, model_id=None):
        self.model_id = model_id
        self.train_data = []
        self.train_attrs = []
        self.valid_data = []
        self.valid_attrs = []
        self.test_data = []
        self.test_complete_attrs = []

    # Get Train Data
    def get_train_data(self):
        datareader = fdata()
        datareader.get_data('train')
        self.train_data = datareader.fmaps_list  # get images
        attrs = datareader.fmaps_attr_list  # get attributes

        # make attributes binary
        # TODO Shuffle for training
        self.train_attrs = []
        for k in attrs:
            if k[1] == 'spoof':
                self.train_attrs.append(0)
            else:
                self.train_attrs.append(1)

    # Get Validation Data
    def get_valid_data(self):
        datareader = fdata()
        datareader.get_data('dev')
        self.valid_data = datareader.fmaps_list  # get images
        attrs = datareader.fmaps_attr_list  # get attributes

        # make attributes binary
        # TODO Shuffle for training
        self.valid_attrs = []
        for k in attrs:
            if k[1] == 'spoof':
                self.valid_attrs.append(0)
            else:
                self.valid_attrs.append(1)

    # Get Test Data
    def get_test_data(self):
        datareader = fdata()
        datareader.get_data('eval')
        self.test_data = datareader.fmaps_list  # get images
        self.test_complete_attrs = datareader.fmaps_attr_list  # get attributes
        self.filenames = datareader.wavfilenames

    def get_wav_filenames(self):
        return self.filenames

    # Get 64 images from a test utterance
    def get_test_utterance(self, utterance):
        ut_images = []
        if utterance not in self.filenames:
            return -1
        # attr = 0 if spoof or 1 if genuine
        attr = 1;
        for i in range(0, len(self.test_complete_attrs)):
            if self.test_complete_attrs[i][0] == utterance:
                ut_images.append(self.test_data[i])
                if self.test_complete_attrs[i][1] == 'spoof':
                    attr = 0;



        # Acquire 64 random images from the soundfile
        # TODO This must be 64 !!
        ut_images = random.sample(ut_images, 64)

        return ut_images , attr

    # X = Features
    def inference(self, X, reuse=True, is_training=True):
        with tf.variable_scope("inference", reuse=reuse):
            # Implement your network here

            # Input Layer
            input_layer = tf.reshape(X, [-1, 17, 64, 1])

            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                inputs=input_layer,
                filters=32,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            # Pooling Layer #1
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

            # Convolutional Layer #2
            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

            # Pooling Layer #2
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

            # Flatten tensor into a batch of vectors
            pool2_flat = tf.reshape(pool2, [-1, 4 * 16 * 64])

            # Dense Layer
            dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

            # # Add dropout operation; 0.6 probability that element will be kept
            # dropout = tf.layers.dropout(
            #     inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN) TODO (BONUS) add dropout

            # Logits layer <- with dropout
            # logits = tf.layers.dense(inputs=dropout, units=10)

            # Logits layer
            logits = tf.layers.dense(inputs=dense, units=2)
            Y = logits

        return Y

    def define_train_operations(self):

        # # --- Train computations
        # Read Train Files
        self.get_train_data()

        self.X_data_train = tf.placeholder(tf.float32)  # Define this TODO train data placeholders
        self.Y_data_train = tf.placeholder(tf.int32)  # Define this TODO do placeholders need self. ?

        self.Y_net_train = self.inference(self.X_data_train, reuse=False)  # Network prediction

        # Loss of train data. Calculate Mean Loss
        self.train_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Y_data_train, logits=self.Y_net_train,
                                                           name='train_loss'))

        # define learning rate decay method 
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = 0.1  # Define it TODO is learning rate correct ?

        # define the optimization algorithm
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)  # Define it TODO Is optimizer correct ? TODO Check more optimizers.. maybe Adam

        trainable = tf.trainable_variables()
        self.update_ops = optimizer.minimize(self.train_loss, var_list=trainable, global_step=global_step)

        # --- Validation computations

        # Read Validation Files
        self.get_valid_data()

        self.X_data_valid = tf.placeholder(tf.float32)  # Define this  TODO valid data placeholders
        self.Y_data_valid = tf.placeholder(tf.int32)  # Define this

        self.Y_net_valid = self.inference(self.X_data_valid, reuse=True)  # Network prediction

        # Loss of validation data
        self.valid_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Y_data_valid, logits=self.Y_net_valid,
                                                           name='valid_loss'))

    def train_epoch(self, sess):
        train_loss = 0
        batch = 0

        total_samples = len(self.train_data)
        batches = total_samples // 256

        # Shuffle Images
        
        c = list(zip(self.train_data, self.train_attrs))
        random.shuffle(c)
        self.train_data, self.train_attrs = zip(*c)

        print("Training Batches")
        for batch in progressbar.progressbar(range(0, batches)):
            index = batch * 256
            mean_loss, _ = sess.run([self.train_loss, self.update_ops],
                                    feed_dict={self.X_data_train: self.train_data[index:index + 256],
                                               self.Y_data_train: self.train_attrs[index:index + 256]})
            if math.isnan(mean_loss):
                print('train cost is NaN')
                break
            train_loss += mean_loss
        if total_samples > 0:
            train_loss /= (batch+1)  # TODO Why is this done ?

        return train_loss

    def valid_epoch(self, sess):
        valid_loss = 0
        batch = 0

        total_samples = len(self.valid_data)
        batches = total_samples // 256

        print("Validate Batches")
        for batch in progressbar.progressbar(range(0, batches)):  # loop through train batches:
            index = batch * 256
            mean_loss = sess.run(self.valid_loss, feed_dict={self.X_data_valid: self.valid_data[index:index + 256],
                                                             self.Y_data_valid: self.valid_attrs[index:index + 256]})
            if math.isnan(mean_loss):
                print('valid cost is NaN')
                break
            valid_loss += mean_loss

        if total_samples > 0:
            valid_loss /= (batch+1)
        return valid_loss

    def train(self, sess):
        start_time = time.clock()

        n_early_stop_epochs = 32  # Define it TODO WTF ???????
        n_epochs = 64  # Define it

        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=4)

        early_stop_counter = 0

        init_op = tf.group(tf.global_variables_initializer())

        sess.run(init_op)

        min_valid_loss = sys.float_info.max
        epoch = 0
        while (epoch < n_epochs):
            print("Training Epoch:" + str(epoch+1)+"/"+str(n_epochs))
            epoch += 1
            epoch_start_time = time.clock()

            train_loss = self.train_epoch(sess)
            valid_loss = self.valid_epoch(sess)

            epoch_end_time = time.clock()

            info_str = 'Epoch=' + str(epoch) + ', Train: ' + str(train_loss) + ', Valid: '
            info_str += str(valid_loss) + ', Time=' + str(epoch_end_time - epoch_start_time)
            print(info_str)

            if valid_loss < min_valid_loss:
                print('Best epoch=' + str(epoch))
                save_variables(sess, saver, epoch, self.model_id)
                min_valid_loss = valid_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter > n_early_stop_epochs:
                # too many consecutive epochs without surpassing the best model
                print('stopping early')
                break

        end_time = time.clock()
        print('Total time = ' + str(end_time - start_time))

    # TODO Predictions
    def define_predict_operations(self):
        self.get_test_data()
        self.X_test_data = tf.placeholder(tf.float32)

        # Pass through model
        self.Y_net_test = self.inference(self.X_test_data, reuse=False)

        #Use softmax
        self.Y_net_test = tf.nn.softmax(self.Y_net_test, axis=1)

        #Use log
        self.Y_net_test = tf.log(self.Y_net_test)

        #Sum Columns
        self.Y_net_test = tf.reduce_sum(self.Y_net_test,0)

        # Y_net_test <= [l0,l1]


    # Predict an utterance, Returns the prediction and the actual attribute
    def predict_utterance(self, sess, wavname, model):
        # saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=4)
        init_op = tf.group(tf.global_variables_initializer())

        sess.run(init_op)
        testdata , attr = self.get_test_utterance(wavname)
        results = sys.float_info.max

        # results <= [l0, l1]
        results = sess.run(self.Y_net_test, feed_dict={self.X_test_data: testdata})

        #Spoof if l0 >= l1
        if results[0] >= results[1]:
            # Spoof = 0
            return 0 , attr
        else:
            # Genuine = 1
            return 1 , attr




