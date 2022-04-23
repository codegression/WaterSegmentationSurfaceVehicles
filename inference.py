#!/usr/bin/env python
"""
Author: Codegression
This module is to perform inference on a trained model.
"""

import architecture
import PIL
import datetime
import numpy as np


model = architecture.create()
model.load_weights('model-weights.h5') #load saved weights

#The first inference is usually slow. So let's perform inference with a dummy input so that subsequent ones will be faster
x_dummy = np.zeros((1, architecture.IMAGE_WIDTH, architecture.IMAGE_HEIGHT, 3), dtype=np.uint8)
model.predict(x_dummy) 

def infer(image, alpha):
    """This function performs inference on an input image and turns a blended image with water pixels color-coded.

    Args:
        image (PIL.Image): input image
        alpha (int): Opacity value from 0 to 255

    Returns:
        PIL.Image : input image with water pixels shaded in blue
        Datetime.timespan : Elapsed processing time
    """
    image = image.convert ("RGBA")
    background = PIL.Image.new('RGBA', image.size, (255, 255, 255))
    image = PIL.Image.alpha_composite(background, image)
    image = image.convert ("RGB")
    original_size = image.size
    inputimage = image.resize((architecture.IMAGE_WIDTH, architecture.IMAGE_HEIGHT), PIL.Image.ANTIALIAS)  
   
    x = np.array(inputimage)          
    x  = np.expand_dims(x, axis=0)
    startinf = datetime.datetime.now()
    
    preds = model.predict(x)  
    endinf = datetime.datetime.now()
    
    preds = np.squeeze(preds, axis=0)
    preds = preds[:,:,0]
    preds = preds * 255
    preds = preds.astype(np.uint8)

    mask_alpha = np.zeros((preds.shape[0], preds.shape[1], 4), dtype=np.uint8)
    mask_alpha[:,:,2] = preds

    image2 = PIL.Image.fromarray(mask_alpha)
    image2 = image2.resize(original_size, PIL.Image.NEAREST)

    y = np.array(image2)   
    y_blue = y[:,:,2]
    y_blue[y_blue > 128] = 255
    y_blue[y_blue < 128] = 0
    y[:,:,2] = y_blue
    y[:,:,3] = (y_blue/255)*alpha
    image2 = PIL.Image.fromarray(y)
    image.paste(image2, (0, 0), image2)

    return image, endinf -startinf