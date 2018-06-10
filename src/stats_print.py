import os
from lib.model_io import *
import matplotlib.pyplot as plt

stats_dir = "saved_model_stats/"
filenames = [x for x in os.listdir(stats_dir) if x.endswith(".pcl")]

for filename in filenames:
    dict = load_model_stats(filename)
    train_losses_plt = plt.plot(dict.get('train_losses'), label='Train Losses')
    valid_losses_plt = plt.plot(dict.get('valid_losses'), label='Valid Losses')
    plt.legend()
    text = "Model: " + dict.get("model") + ", Epochs: " + str(dict.get("epochs")) + ", Early_Stop: " + str(dict.get("early_stop"))
    text += "\nLearning Rate: " + str(dict.get("learning_rate")) + ", Batch Normalization: " + str(dict.get("batch_norm")) + ", Accuracy: " + str(dict.get("accuracy"))
    plt.title(text)
    plt.xlabel("Epochs")
    plt.ylabel("Losses")

    plt.show()
