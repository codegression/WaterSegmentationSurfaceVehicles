#!/usr/bin/env python
"""
Author: Codegression
This module is to perform training
"""


import numpy as np
import architecture
import data
from skimage.io import imshow
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"




def train():
    """_Performs training and saves the trained model
    """
    np.random.seed = architecture.RANDOM_SEED
    model = architecture.create()

    #load data
    X_train, Y_train = data.load_tampere(architecture.IMAGE_WIDTH, architecture.IMAGE_HEIGHT, architecture.IMAGE_CHANNEL)

    #imshow(X_train[11])
    #plt.show()

    model.fit(X_train, Y_train, validation_split=0.1, epochs=100)
    #model.save("model.h5")
    model.save_weights("model-weights.h5")

if __name__ == '__main__':
    train()