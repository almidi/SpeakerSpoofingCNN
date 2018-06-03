import os
import tensorflow as tf
import progressbar
from src.model import CNN
from lib.model_io import get_modle_id
from lib.model_io import restore_variables

print("Get Model ID")
model_id = get_modle_id()


# Create the network
print("Create Network")
network = CNN(model_id)

print("Define Predict Operations")
network.define_predict_operations()

# Recover the parameters of the model
print("Recover the parameters of the model")
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

#Restore Variables
print("Restore Variables")
restore_variables(sess)

# Iterate through eval files and calculate the classification scores
filenames = network.get_wav_filenames()

score = 0

for filename in progressbar.progressbar(filenames) :
    prediction , attr = network.predict_utterance(sess, filename, "")
    if prediction == attr : score += 1

print("\n" + str((score/len(filenames))*100) + "% Success Rate" )

sess.close()
        

