#!/usr/bin/env python
"""
Author: Codegression
This module is to plot the model
"""


import numpy as np
import architecture
import data
from skimage.io import imshow
import matplotlib.pyplot as plt
import os
import visualkeras
from tensorflow import keras
from tensorflow.keras import layers, models
from PIL import ImageFont
import pydotplus
#import graphviz

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"




def plot():
    """_Performs plotting
    """

    #model = architecture.create()
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.summary()
    visualkeras.layered_view(model) 
   

model = architecture.create()
model.summary()
#a = visualkeras.layered_view(model, legend=True) 
#a.save('test.png')
#keras.utils.vis_utils.pydot = pydotplus
#keras.utils.plot_model(model, 'test2.png')