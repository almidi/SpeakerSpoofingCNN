import os
import tensorflow as tf
import progressbar
from src.model import CNN
from lib.model_io import *

# TODO These values should be restored from the model
def predict(epochs=12, model='baseline', early_stop=4, learning_rate=0.0001, batch_norm=True, batch_size=256):
    #Enable gpu memory growth
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #Get Model ID
    print("Predicting")
    model_id = get_model_id()
    print("Model ID: "+str(model_id))

    # # Get Model Id Stats
    # dic = load_model_stats(str(model_id)+"-stats.pcl")

    # Create the network
    print("Create Network")
    network = CNN(model_id, model=model, epochs=epochs, early_stop=early_stop, learning_rate=learning_rate,
                  batch_norm=batch_norm, batch_size=batch_size)

    print("Define Predict Operations")
    network.define_predict_operations()

    # Recover the parameters of the model
    print("Recover the parameters of the model")
    sess = tf.Session(config=config)

    #Restore Variables
    print("Restore Variables")
    restore_variables(sess)

    # Iterate through eval files and calculate the classification scores
    filenames = network.get_wav_filenames()

    score = 0

    for filename in progressbar.progressbar(filenames) :
        prediction , attr = network.predict_utterance(sess, filename)
        if prediction == attr : score += 1

    final_score = (score/len(filenames))*100
    print("\n" + str(final_score) + "% Success Rate" )

    sess.close()
    return final_score

if __name__ == "__main__":
    predict()
