from __future__ import division, print_function

import os
import argparse
import json
import librosa
import numpy as np
import tensorflow as tf

__author__ = """Vassilis Tsiaras (tsiaras@csd.uoc.gr)"""
#    Vassilis Tsiaras <tsiaras@csd.uoc.gr>
#    Computer Science Department, University of Crete.



def read_model_id(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as fid:
            model_id = int(fid.read())
        fid.close()
    else:
        write_model_id(filename, 1)
        model_id = 0
        
    return model_id      


def write_model_id(filename, model_id):
    model_id_txt = str(model_id) 
    with open(filename, 'w') as fid:
        fid.write(model_id_txt)
    fid.close() 

def get_modle_id():
    model_id_filename = #Define it
    model_id = read_model_id(model_id_filename) + 1 # Reserve the next model_id. If file does not exists then create it 
    write_model_id(model_id_filename, model_id) 

    return model_id 

def save_variables(sess, saver, epoch, model_id): 
    model_path = # Define it
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    checkpoint_path = os.path.join(model_path, 'cnn')
    saver.save(sess, checkpoint_path, global_step=epoch)

def restore_variables(sess):
    variables_to_restore = {
    var.name[:-2]: var for var in tf.trainable_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    model_path = # Define it
    print(model_path)   
    ckpt = tf.train.get_checkpoint_state(model_path)  
    print(ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path) 


  

    





