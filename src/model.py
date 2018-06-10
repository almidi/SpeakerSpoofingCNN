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


# from lib.precision import _FLOATX


class CNN(object):

    def __init__(self, model_id=None, model='baseline', batch_size=256, learning_rate=0.0001, epochs=12, early_stop=4, batch_norm=True, mean = 0,std = 1):
        self.model_id = model_id
        self.train_data = []
        self.train_attrs = []
        self.valid_data = []
        self.valid_attrs = []
        self.test_data = []
        self.mean = mean
        self.std = std
        self.test_complete_attrs = []
        self.BATCH_SIZE = batch_size
        self.LEARNING_RATE = learning_rate
        self.EPOCHS = epochs
        self.EARLY_STOP = early_stop
        self.MODEL = model
        self.BATCH_NORM = batch_norm

    def reset(self):
        tf.reset_default_graph
    # Get Train Data
    def get_train_data(self):
        datareader = fdata()
        datareader.get_data('train')
        self.train_data = datareader.fmaps_list  # get images
        self.train_data = np.array(self.train_data, np.float32)

        #get mean
        self.mean = np.mean(self.train_data)
        #get standard deviation
        self.std = np.std(self.train_data)

        #normalize dataset
        print("Normalizing Train Data:")
        for i in  progressbar.progressbar(range(0,len(self.train_data)))
            self.train_data[i] = np.subtract(self.train_data[i],mean)
            self.train_data[i] = np.divide(self.train_data[i],std)

        attrs = datareader.fmaps_attr_list  # get attributes

        # make attributes binary
        # TODO Shuffle for training
        self.train_attrs = []
        for k in attrs:
            if k[1] == 'spoof':
                self.train_attrs.append(0)
            else:
                self.train_attrs.append(1)

        self.train_attrs = np.array(self.train_attrs, np.int32)

    # Get Validation Data
    def get_valid_data(self):
        datareader = fdata()
        datareader.get_data('dev')
        self.valid_data = datareader.fmaps_list  # get images
        self.valid_data = np.array(self.valid_data, np.float32)

        #normalize dataset
        print("Normalizing Valid Data:")
        for i in progressbar.progressbar(range(0,len(self.valid_data)))
            self.valid_data[i] = np.subtract(self.valid_data[i],mean)
            self.valid_data[i] = np.divide(self.valid_data[i],std)

        attrs = datareader.fmaps_attr_list  # get attributes

        # make attributes binary
        # TODO Shuffle for training
        self.valid_attrs = []
        for k in attrs:
            if k[1] == 'spoof':
                self.valid_attrs.append(0)
            else:
                self.valid_attrs.append(1)

        self.valid_attrs = np.array(self.valid_attrs, np.int32)

    # Get Test Data
    def get_test_data(self):
        datareader = fdata()
        datareader.get_data('eval')
        self.test_data = datareader.fmaps_list  # get images

        print("Normalizing Valid Data:")
        for i in progressbar.progressbar(range(0,len(self.test_data)))
            self.test_data = np.subtract(self.test_data,mean)
            self.test_data = np.divide(self.test_data,std)

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

        return ut_images, attr

    # X = Features
    def vd10fdInference(self, X, reuse=True, is_training=True):
        with tf.variable_scope("inference", reuse=reuse):
            # Implement your network here
            ##################################################### vd10-fpad-dpad ###############################################

            conv1 = tf.layers.conv2d(
                inputs=X,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            conv2 = tf.layers.conv2d(
                inputs=conv1,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            pool1 = tf.nn.max_pool(value=conv2, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')

            conv3 = tf.layers.conv2d(
                inputs=pool1,
                filters=128,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            conv4 = tf.layers.conv2d(
                inputs=conv3,
                filters=128,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            pool2 = tf.nn.max_pool(value=conv4, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')

            conv5 = tf.layers.conv2d(
                inputs=pool2,
                filters=128,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            conv6 = tf.layers.conv2d(
                inputs=conv5,
                filters=128,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            pool3 = tf.nn.max_pool(value=conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            conv7 = tf.layers.conv2d(
                inputs=pool3,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            conv8 = tf.layers.conv2d(
                inputs=conv7,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            pool4 = tf.nn.max_pool(value=conv8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            conv9 = tf.layers.conv2d(
                inputs=pool4,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            conv10 = tf.layers.conv2d(
                inputs=conv9,
                filters=256,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            pool5 = tf.nn.max_pool(value=conv10, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

            # Flatten tensor into a batch of vectors
            flat = tf.layers.flatten(pool5)

            # Dense Layer with batch normalization TODO How many dense layers ?
            if self.BATCH_NORM :
                norm_flat = tf.layers.batch_normalization(flat, training=is_training)
                dense = tf.layers.dense(inputs=norm_flat, units=2, activation=tf.nn.relu)
            else :
                # Dense Layer TODO How many dense layers ?
                dense = tf.layers.dense(inputs=flat, units=2, activation=tf.nn.relu)

            #  Add dropout operation; 0.6 probability that element will be kept
            dropout = tf.layers.dropout(
                inputs=dense, rate=0.4, training=is_training)

            # Logits layer <- with dropout
            logits = tf.layers.dense(inputs=dropout, units=10)

            # Logits layer
            # logits = dense
            Y = logits

        return Y

    def baselineInference(self, X, reuse=True, is_training=True):
        with tf.variable_scope("inference", reuse=reuse):
            # Implement your network here

            #############################################################BASELINE########################################################
            # Convolutional Layer #1
            conv1 = tf.layers.conv2d(
                inputs=X,
                filters=32,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            pool1 = tf.nn.max_pool(value=conv1, ksize=[1, 2, 1, 1], strides=[1, 2, 1, 1], padding='VALID')

            conv2 = tf.layers.conv2d(
                inputs=pool1,
                filters=64,
                kernel_size=[3, 3],
                padding="same",
                activation=tf.nn.relu)

            pool2 = tf.nn.max_pool(value=conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
            # Flatten tensor into a batch of vectors
            flat = tf.layers.flatten(pool2)

            # Dense Layer with batch normalization TODO How many dense layers ?
            if self.BATCH_NORM :
                norm_flat = tf.layers.batch_normalization(flat, training=is_training)
                dense = tf.layers.dense(inputs=norm_flat, units=2, activation=tf.nn.relu)
            else :
                dense = tf.layers.dense(inputs=flat, units=2, activation=tf.nn.relu)

            #  Add dropout operation; 0.6 probability that element will be kept
            dropout = tf.layers.dropout(
                inputs=dense, rate=0.4, training=is_training)

            # Logits layer <- with dropout
            logits = tf.layers.dense(inputs=dropout, units=10)

            # Logits layer
            # logits = dense
            Y = logits

        return Y

    def define_train_operations(self):

        # # --- Train computations
        # Read Train Files
        self.get_train_data()

        self.X_data_train = tf.placeholder(tf.float32,
                                           shape=[self.BATCH_SIZE, 64, 17,
                                                  1])  # Define this TODO train data placeholders
        self.Y_data_train = tf.placeholder(tf.int32)  # Define this TODO do placeholders need self. ?

        if self.MODEL == 'baseline':
            self.Y_net_train = self.baselineInference(self.X_data_train, reuse=False)  # Network prediction
        elif self.MODEL == 'vd10fd':
            self.Y_net_train = self.vd10fdInference(self.X_data_train, reuse=False)  # Network prediction
        else:
            raise NameError('No Model Named ' + self.MODEL)

        # Loss of train data. Calculate Mean Loss
        train_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Y_data_train,
                                                                             logits=self.Y_net_train,
                                                                             name='train_loss')

        self.train_loss = tf.reduce_mean(train_cross_entropy)

        # define learning rate decay method
        global_step = tf.Variable(0, trainable=False, name='global_step')

        # define the optimization algorithm
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.LEARNING_RATE)  # Define it TODO Is optimizer correct ? TODO Check more optimizers..

        trainable = tf.trainable_variables()
        self.update_ops = optimizer.minimize(self.train_loss, var_list=trainable, global_step=global_step)

        # --- Validation computations

        # Read Validation Files
        self.get_valid_data()

        self.X_data_valid = tf.placeholder(tf.float32,
                                           shape=[self.BATCH_SIZE, 64, 17, 1])  # Define this  TODO valid data placeholders

        self.Y_data_valid = tf.placeholder(tf.int32)  # Define this

        if self.MODEL == 'baseline':
            self.Y_net_valid = self.baselineInference(self.X_data_valid, reuse=True)  # Network prediction
        elif self.MODEL == 'vd10fd':
            self.Y_net_valid = self.vd10fdInference(self.X_data_valid, reuse=True)  # Network prediction
        else:
            raise NameError('No Model Named ' + self.MODEL)

        # Loss of validation data
        valid_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.Y_data_valid,
                                                                             logits=self.Y_net_valid,
                                                                             name='valid_loss')
        self.valid_loss = tf.reduce_mean(valid_cross_entropy)

    def train_epoch(self, sess):
        train_loss = 0
        batch = 0

        total_samples = len(self.train_data)
        batches = total_samples // self.BATCH_SIZE

        # Shuffle Images

        c = list(zip(self.train_data, self.train_attrs))
        random.shuffle(c)
        self.train_data, self.train_attrs = zip(*c)

        print("Training Batches")
        for batch in progressbar.progressbar(range(0, batches)):
            index = batch * self.BATCH_SIZE

            data_batch = self.train_data[index:index + self.BATCH_SIZE]
            data_batch = np.reshape(data_batch, [self.BATCH_SIZE, 64, 17, 1])
            data_batch = np.array(data_batch, np.float32)

            attr_batch = self.train_attrs[index:index + self.BATCH_SIZE]

            mean_loss, _ = sess.run([self.train_loss, self.update_ops],
                                    feed_dict={self.X_data_train: data_batch,
                                               self.Y_data_train: attr_batch})
            if math.isnan(mean_loss):
                print('train cost is NaN')
                break
            train_loss += mean_loss
        if total_samples > 0:
            train_loss /= batch  # TODO Why is this done ?

        return train_loss

    def valid_epoch(self, sess):
        valid_loss = 0
        batch = 0

        total_samples = len(self.valid_data)
        batches = total_samples // self.BATCH_SIZE

        print("Validate Batches")
        for batch in progressbar.progressbar(range(0, batches)):  # loop through train batches:
            index = batch * self.BATCH_SIZE

            data_batch = self.valid_data[index:index + self.BATCH_SIZE]
            data_batch = np.reshape(data_batch, [self.BATCH_SIZE, 64, 17, 1])
            attr_batch = self.valid_attrs[index:index + self.BATCH_SIZE]

            mean_loss = sess.run(self.valid_loss, feed_dict={self.X_data_valid: data_batch,
                                                             self.Y_data_valid: attr_batch})
            if math.isnan(mean_loss):
                print('valid cost is NaN')
                break
            valid_loss += mean_loss

        if total_samples > 0:
            valid_loss /= batch
        return valid_loss

    def train(self, sess):
        start_time = time.clock()

        losses = []

        n_early_stop_epochs = self.EARLY_STOP  # Define it TODO WTF ???????
        n_epochs = self.EPOCHS  # Define it

        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=4)

        early_stop_counter = 0

        init_op = tf.group(tf.global_variables_initializer())

        sess.run(init_op)

        min_valid_loss = sys.float_info.max
        epoch = 0
        while (epoch < n_epochs):
            print("Training Epoch:" + str(epoch + 1) + "/" + str(n_epochs))
            epoch += 1
            epoch_start_time = time.clock()

            train_loss = self.train_epoch(sess)
            valid_loss = self.valid_epoch(sess)

            epoch_end_time = time.clock()

            info_str = 'Epoch=' + str(epoch) + ', Train: ' + str(train_loss) + ', Valid: '
            info_str += str(valid_loss) + ', Time=' + str(epoch_end_time - epoch_start_time)
            print(info_str)

            losses.append([train_loss,valid_loss])

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
        return losses,self.mean,self.std;

    # TODO Predictions
    def define_predict_operations(self):
        self.get_test_data()
        self.X_test_data = tf.placeholder(tf.float32, shape=[None, 64, 17, 1])

        # Pass through model
        print("Predicting " + self.MODEL + " model !!")
        if self.MODEL == 'baseline':
            self.Y_net_test = self.baselineInference(self.X_test_data, reuse=True, is_training=False)
        elif self.MODEL == 'vd10fd':
            self.Y_net_test = self.vd10fdInference(self.X_test_data, reuse=True, is_training=False)
        else:
            raise NameError('No Model Named ' + self.MODEL)

        # Use softmax
        self.Y_net_test = tf.nn.softmax(self.Y_net_test, axis=1)

        # Use log
        self.Y_net_test = tf.log(self.Y_net_test)

        # Sum Columns
        self.Y_net_test = tf.reduce_sum(self.Y_net_test, 0)

        # Y_net_test <= [l0,l1]

    # Predict an utterance, Returns the prediction and the actual attribute
    def predict_utterance(self, sess, wavname):
        # saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=4)
        init_op = tf.group(tf.global_variables_initializer())

        sess.run(init_op)
        testdata, attr = self.get_test_utterance(wavname)
        testdata = np.reshape(testdata, [64, 64, 17, 1])
        results = sys.float_info.max

        # results <= [l0, l1]
        results = sess.run(self.Y_net_test, feed_dict={self.X_test_data: testdata})

        # Spoof if l0 >= l1
        if results[0] >= results[1]:
            # Spoof = 0
            return 0, attr
        else:
            # Genuine = 1
            return 1, attr
