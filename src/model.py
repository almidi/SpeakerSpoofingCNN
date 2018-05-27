from __future__ import division, print_function

import sys
import time
from datetime import datetime
import math
import tensorflow as tf
from lib.model_io import save_variables
from lib.precision import _FLOATX


class CNN(object):

    def __init__(self, model_id=None):
        self.model_id = model_id


    def inference(self, X, reuse=True, is_training=True):
        with tf.variable_scope("inference", reuse=reuse):
            # Implement your network here

        return Y 

    def define_train_operations(self):
 
        # --- Train computations 
        self.trainDataReader = trainDataReader

        X_data_train = tf.placeholder( .... ) # Define this  

        Y_data_train = tf.placeholder( .... ) # Define this  

        Y_net_train = self.inference(X_data_train, reuse=False) # Network prediction


         # Loss of train data
        self.train_loss = tf.reduce_mean(tf.nn.sparce_softmax_cross_entropy_with_logits(labels=Y_data_train, logits=Y_net_train, name='train_loss'))   


        # define learning rate decay method 
        global_step = tf.Variable(0, trainable=False, name='global_step')
        learning_rate = .... # Define it

        # define the optimization algorithm
        optimizer = .... # Define it

        trainable = tf.trainable_variables()
        self.update_ops = optimizer.minimize(self.train_loss, var_list=trainable, global_step=global_step)

        # --- Validation computations
        X_data_valid = tf.placeholder( .... ) # Define this  
        Y_data_valid = tf.placeholder( .... ) # Define this  


        Y_net_valid = self.inference(X_data_valid, reuse=True) # Network prediction
     
        # Loss of validation data
        self.valid_loss = tf.reduce_mean(tf.nn.sparce_softmax_cross_entropy_with_logits(labels=Y_data_valid, logits=Y_net_valid, name='valid_loss'))

 

    def train_epoch(self, sess):
        train_loss = 0
        total_batches = 0 
        
        while #loop through train batches:
            mean_loss, _ = sess.run([self.train_loss, self.update_ops], feed_dict=......) 
            if math.isnan(mean_loss):
                print('train cost is NaN')
                break
            train_loss += mean_loss
            total_batches += 1  
        
        if total_samples > 0:  
            train_loss /= total_batches   

        return train_loss 
        

    def valid_epoch(self, sess):
        valid_loss = 0
        total_batches = 0 

        while # Loop through valid batches:
            mean_loss = sess.run(self.valid_loss, feed_dict=....)
            if math.isnan(mean_loss):
                print('valid cost is NaN')
                break
            valid_loss += mean_loss
            total_batches += 1  

        if total_samples > 0:  
            valid_loss /= total_batches   

        return valid_loss


    def train(self, sess):
        start_time = time.clock()

        n_early_stop_epochs = # Define it
        n_epochs = # Define it

        saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=4)

        early_stop_counter = 0

        init_op = tf.group(tf.global_variables_initializer())

        sess.run(init_op) 

        min_valid_loss = sys.float_info.max
        epoch = 0
        while (epoch < n_epochs):
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


    def define_predict_operations(self):
        self.X_data_test_placeholder = tf.placeholder(....)

        self.Y_net_test = self.inference(self.X_data_test_placeholder, reuse=False)


    def predict_utterance(self, sess, testDataReader, dataWriter):
        # Define it

         
           
        


