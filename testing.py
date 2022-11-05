#!/usr/bin/env python
"""
Author: Codegression
This module is to perform testing
"""


import numpy as np
import architecture
import data
import matplotlib.pyplot as plt
import os
import pickle
import keras.preprocessing.image
import tensorflow as tf
import sklearn.metrics 


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"




def test():
    """_Performs testing the trained model on unseen data
    It saves Y_pre and Y_act so that visualisation can be done using Jupyter notebook
    """
    np.random.seed = architecture.RANDOM_SEED
    model = architecture.create()
    
    tf.random.set_seed(architecture.RANDOM_SEED)  
    
    """
    Steps
    1. Load model
    2. Load test data
    3. Concatenate data
    4. Test the model
    5. Flatten prediction results
    6. Get statistics
    7. Save results for visualisation using Jupyter Notebook
   """
   
    #Load model
   
    model = architecture.create()
    model.load_weights('dump/model-weights.h5') #load saved weights
    #Load test data
    X1, Y1 = data.load_intcatchtest(image_width=architecture.IMAGE_WIDTH, 
                              image_height=architecture.IMAGE_HEIGHT, 
                              image_channel=architecture.IMAGE_CHANNEL, 
                              dataset_path = '../datasets/IntCatch dataset/water_segmentation_test',
                              number_of_samples = 87, 
                              x_filename = 'x.npy',
                              y_filename = 'y.npy')
    
    X2, Y2 = data.load_tampere(train=False,
                         image_width=architecture.IMAGE_WIDTH,
                         image_height=architecture.IMAGE_HEIGHT,
                         image_channel=architecture.IMAGE_CHANNEL,
                         dataset_path = '../datasets/Tampere-WaterSeg',
                         number_of_samples = 200,
                         x_filename = 'x_test.npy',
                         y_filename = 'y_test.npy')
    

    #Concatenate
    X = np.concatenate((X1, X2), axis=0)
    Y = np.concatenate((Y1, Y2), axis=0)    
    
    #Prediction    
    Y_hat = model.predict(X)
    
    #Flatten prediction results
    Y = Y.flatten()
    Y_hat = Y_hat.flatten()
    
    Y = Y.astype(int)
    Y_hat = Y_hat.astype(int)
    
    #Get statistics    
    print('Train report', sklearn.metrics.classification_report(Y, Y_hat))
    print('Train conf matrix', sklearn.metrics.confusion_matrix(Y, Y_hat))
    
    #Save predictions    
    with open('dump/true_Y_test.npy', 'wb') as f:
        np.save(f, Y)
    with open('dump/actual_Y_test.npy', 'wb') as f:
        np.save(f, Y_hat)
      
    print("Results saved.")

if __name__ == '__main__':
    test()