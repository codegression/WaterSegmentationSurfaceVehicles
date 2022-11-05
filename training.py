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
import pickle
import keras.preprocessing.image
import tensorflow as tf
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"




def train():
    """_Performs training and saves the trained model
    """
    np.random.seed = architecture.RANDOM_SEED
    model = architecture.create()
    
    tf.random.set_seed(architecture.RANDOM_SEED)
   
    """
    Steps
    1. Use ImageNet weights up to 177th layer 
    2. Train with general water segmentation dataset and freeze the weights up to 177th layer
    3. Concatenate tampere and IntCatch train dataset
    4. Train on the merged dataset from 177th layer onwards     
    5. Get test accuracy on Intcatch test dataset
        
    """

    for i in range(177):  
        model.layers[i].trainable = False
        


    #watersegmentation dataset
    
    if not os.path.exists("dump/model-weights-firstpass.h5"):
        X, Y = data.load_watersegmentation(architecture.IMAGE_WIDTH, 
                                           architecture.IMAGE_HEIGHT, 
                                           architecture.IMAGE_CHANNEL)
           
        history = model.fit(X, Y, validation_split=0.1, epochs=30)
        with open('dump/history1.pkl', 'wb') as file:
           pickle.dump(history.history, file)
           
        model.save_weights("dump/model-weights-firstpass.h5")
    else:
        model.load_weights("dump/model-weights-firstpass.h5")
        print("Loaded first pass weights")

    
    
    #Tampere and IntCatch train dataset
    for i in range(198):
        model.layers[i].trainable = False
    
    X1, Y1 = data.load_tampere(train=True, 
                               image_width=architecture.IMAGE_WIDTH, 
                               image_height=architecture.IMAGE_HEIGHT,
                               image_channel=architecture.IMAGE_CHANNEL,
                               dataset_path = '../datasets/Tampere-WaterSeg',
                               number_of_samples = 800,
                               x_filename = 'x_train.npy',
                               y_filename = 'y_train.npy')
    
    X2, Y2 = data.load_intcatch(image_width=architecture.IMAGE_WIDTH, 
                                image_height=architecture.IMAGE_HEIGHT, 
                                image_channel=architecture.IMAGE_CHANNEL, 
                                dataset_path = '../datasets/IntCatch dataset/water_segmentation_training',
                                number_of_samples = 191*2, 
                                x_filename = 'x.npy', 
                                y_filename = 'y.npy')
    
    X3, Y3 = data.load_intcatch(image_width=architecture.IMAGE_WIDTH, 
                                image_height=architecture.IMAGE_HEIGHT, 
                                image_channel=architecture.IMAGE_CHANNEL, 
                                dataset_path = '../datasets/IntCatch dataset/water_segmentation_validation',
                                number_of_samples = 80, 
                                x_filename = 'x.npy', 
                                y_filename = 'y.npy')
    
    X = np.concatenate((X1, X2, X3), axis=0)
    Y = np.concatenate((Y1, Y2, Y3), axis=0)    
    history = model.fit(X, Y, validation_split=0.1, epochs=100)
    
    with open('dump/history2.pkl', 'wb') as file:
       pickle.dump(history.history, file)

  
    model.save_weights("dump/model-weights.h5")

if __name__ == '__main__':
    train()