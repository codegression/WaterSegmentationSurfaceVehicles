#!/usr/bin/env python
"""
Author: Codegression
This module is used to load data from datasets
"""


import os
import numpy as np
from tqdm import tqdm 
from skimage.io import imread, imshow, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt
      



def load_intcatchtest(image_width=224, image_height=224, image_channel=3, dataset_path = '../datasets/IntCatch dataset/water_segmentation_test', number_of_samples = 87, x_filename = 'x_test.npy', y_filename = 'y_test.npy'):
    """_summary_

    Args:
        image_width (int, optional): width of image. Defaults to 224.
        image_height (int, optional): height of image. Defaults to 224.
        image_channel (int, optional): image channel. Defaults to 3.
        dataset_path (str, optional):database path. Defaults to '../datasets/IntCatch dataset/water_segmentation_test'.
        number_of_samples (int, optional): number of samples. Defaults to 600.
        x_filename (str, optional): file name to save X. Defaults to 'x.npy'.
        y_filename (str, optional): file name to save Y. Defaults to 'y.npy'.

    Returns:
        np array: X
        np array: Y
    """
    print("Loading IntCatch test dataset with the following parameters")
    print(locals())
    
    if not os.path.exists(dataset_path + "/" + x_filename) or not os.path.exists(dataset_path + "/" + y_filename):
        folders = next(os.walk(dataset_path))[1]
        X = np.zeros((number_of_samples, image_height, image_width, image_channel), dtype=np.uint8)
        Y = np.zeros((number_of_samples, image_height, image_width, 1), dtype=bool)
        count = 0
        
        files = next(os.walk(dataset_path))[2]   
        for file in files:       
            
            if os.path.splitext(file)[1]!='.png':
                continue
    
            name =  os.path.splitext(file)[0]
            
            if "annotation" in name:
                continue
            
            print(str(count) + " " + file)
            
            ####### Image X ################################
          
            img = imread(dataset_path + "/" + file)[:,:,:image_channel]  
            img = resize(img, (image_height, image_width), mode='constant', preserve_range=True)
            X[count] = img
            
            
            ####### Image Y ################################
            annotation_filename = dataset_path + "/" + name + "-annotation.png"
                   
            img = imread(annotation_filename, as_gray=True)           
            img = 255 * img
            img = img.astype(np.uint8)
            img[img>0]=255
            img = np.expand_dims(resize(img, (image_height, image_width), mode='constant', preserve_range=True), axis=-1)
            Y[count] = img
                    
            count = count + 1
        
        np.save(dataset_path + "/" + x_filename, X)
        np.save(dataset_path + "/" + y_filename, Y)
    else:
        X = np.load(dataset_path + "/" + x_filename)
        Y = np.load(dataset_path + "/" + y_filename)
        
    return X, Y

def load_intcatch(image_width=224, image_height=224, image_channel=3, dataset_path = '../datasets/IntCatch dataset/water_segmentation_training', number_of_samples = 191*2, x_filename = 'x.npy', y_filename = 'y.npy'):
    """_summary_

    Args:
        image_width (int, optional): width of image. Defaults to 224.
        image_height (int, optional): height of image. Defaults to 224.
        image_channel (int, optional): image channel. Defaults to 3.
        dataset_path (str, optional):database path. Defaults to '../datasets/IntCatch dataset/water_segmentation_training'.
        number_of_samples (int, optional): number of samples. Defaults to 600.
        x_filename (str, optional): file name to save X. Defaults to 'x.npy'.
        y_filename (str, optional): file name to save Y. Defaults to 'y.npy'.

    Returns:
        np array: X
        np array: Y
    """
    print("Loading IntCatch training dataset with the following parameters")
    print(locals())
    
    if not os.path.exists(dataset_path + "/" + x_filename) or not os.path.exists(dataset_path + "/" + y_filename):
        folders = next(os.walk(dataset_path))[1]
        X = np.zeros((number_of_samples, image_height, image_width, image_channel), dtype=np.uint8)
        Y = np.zeros((number_of_samples, image_height, image_width, 1), dtype=bool)
        count = 0
        
        files = next(os.walk(dataset_path))[2]   
        for file in files:       
            
            if os.path.splitext(file)[1]!='.png':
                continue
    
            name =  os.path.splitext(file)[0]
            
            if "annotation" in name:
                continue
            
            print(str(count) + " " + file)
            
            ####### Image X ################################
          
            img = imread(dataset_path + "/" + file)[:,:,:image_channel]  
            img = resize(img, (image_height, image_width), mode='constant', preserve_range=True)
            X[count] = img
            
            
            ####### Image Y ################################
            annotation_filename = dataset_path + "/" + name + "-annotation.png"
                   
            img_m = imread(annotation_filename, as_gray=True)
            img_m = 255 * img_m
            img_m = img_m.astype(np.uint8)
            img_m = np.expand_dims(resize(img_m, (image_height, image_width), mode='constant', preserve_range=True), axis=-1)
            Y[count] = img_m
                    
            count = count + 1
            
            # Flip images
            
            X[count] = img[:, ::-1]
            Y[count] = img_m[:, ::-1]
            count = count + 1
            
        
        np.save(dataset_path + "/" + x_filename, X)
        np.save(dataset_path + "/" + y_filename, Y)
    else:
        X = np.load(dataset_path + "/" + x_filename)
        Y = np.load(dataset_path + "/" + y_filename)
        
    return X, Y



def load_watersegmentation(image_width=224, image_height=224, image_channel=3, dataset_path = '../datasets/Water segmentation/water_v2/water_v2', number_of_samples = 4413*2, x_filename = 'x.npy', y_filename = 'y.npy'):
    """_summary_

    Args:
        image_width (int, optional): width of image. Defaults to 224.
        image_height (int, optional): height of image. Defaults to 224.
        image_channel (int, optional): image channel. Defaults to 3.
        dataset_path (str, optional):database path. Defaults to '../datasets/Water segmentation'.
        number_of_samples (int, optional): number of samples. Defaults to 600.
        x_filename (str, optional): file name to save X. Defaults to 'x.npy'.
        y_filename (str, optional): file name to save Y. Defaults to 'y.npy'.

    Returns:
        np array: X
        np array: Y
    """
    print("Loading water segmentation dataset with the following parameters")
    print(locals())
    if not os.path.exists(dataset_path + "/" + x_filename) or not os.path.exists(dataset_path + "/" + y_filename):
        folders = next(os.walk(dataset_path + "/JPEGImages"))[1]
        X = np.zeros((number_of_samples, image_height, image_width, image_channel), dtype=np.uint8)
        Y = np.zeros((number_of_samples, image_height, image_width, 1), dtype=bool)
        count = 0
        
        
        for folder in folders:
            files = next(os.walk(dataset_path + "/JPEGImages/" + folder))[2]       
            lastannotation = ""
            
            for file in files:
                print(str(count) + " " + file)
                
                ####### Image X ################################
                filename = dataset_path + "/JPEGImages/" + folder + "/" + file
                
                img = imread(filename)[:,:,:image_channel]  
                img = resize(img, (image_height, image_width), mode='constant', preserve_range=True)
                X[count] = img
                
                #imsave("img1.png", img)
                
            
                
                ####### Image Y ################################
                annotation_filename = dataset_path + "/Annotations/" + folder + "/" + file
                annotation_filename = os.path.splitext(annotation_filename)[0] + ".png"
                
                if os.path.exists(annotation_filename):
                    lastannotation = annotation_filename
                elif lastannotation != "":
                    annotation_filename = lastannotation
                
                img_m = imread(annotation_filename, as_gray=True)
                img_m = 255 * img_m
                img_m = img_m.astype(np.uint8)
                img_m = np.expand_dims(resize(img_m, (image_height, image_width), mode='constant', preserve_range=True), axis=-1)
                Y[count] = img_m
                
                #imsave("img_m1.png", img_m)
                count = count + 1
                #################################################
                                
                #Flip image horizontally               
                #imsave("img2.png", img)
                #imsave("img_m2.png", img_m)
                X[count] = img[:, ::-1]
                Y[count] = img_m[:, ::-1]
                count = count + 1
                
               
                
        np.save(dataset_path + "/" + x_filename, X)
        np.save(dataset_path + "/" + y_filename, Y)
    else:
        X = np.load(dataset_path + "/" + x_filename)
        Y = np.load(dataset_path + "/" + y_filename)
        
    return X, Y

def load_tampere(train=True, image_width=224, image_height=224, image_channel=3, dataset_path = '../datasets/Tampere-WaterSeg', number_of_samples = 800, x_filename = 'x.npy', y_filename = 'y.npy'):
    """_summary_

    Args:
        image_width (int, optional): width of image. Defaults to 224.
        image_height (int, optional): height of image. Defaults to 224.
        image_channel (int, optional): image channel. Defaults to 3.
        dataset_path (str, optional):database path. Defaults to '../datasets/Tampere-WaterSeg'.
        number_of_samples (int, optional): number of samples. Defaults to 600.
        x_filename (str, optional): file name to save X. Defaults to 'x.npy'.
        y_filename (str, optional): file name to save Y. Defaults to 'y.npy'.

    Returns:
        np array: X
        np array: Y
    """
    print("Loading Tempere dataset with the following parameters")
    print(locals())
    folders = next(os.walk(dataset_path))[1]    
    data_folders = [x for x in folders if not x.endswith('_mask')]
    mask_folders = [x for x in folders if x.endswith('_mask')]
    X = np.zeros((number_of_samples, image_height, image_width, image_channel), dtype=np.uint8)
    Y = np.zeros((number_of_samples, image_height, image_width, 1), dtype=bool)

    if not os.path.exists(dataset_path + "/" + x_filename):
        count = 0
        for i in range(3):
            if train==True:
                if data_folders[i]=="channel":  #Reserve the "channel" shots for testing
                    continue
            else:
                if data_folders[i]!="channel":  
                    continue
            
            files = next(os.walk(dataset_path + "/" + data_folders[i]))[2]
            for file in files:
                print(str(count) + " " + file)
                img = imread(dataset_path + "/" + data_folders[i] + "/" + file)[:,:,:image_channel]  
                img = resize(img, (image_height, image_width), mode='constant', preserve_range=True)
                X[count] = img
                count = count + 1
                
                if train==True:
                    #Flip image horizontally
                    X[count] = img[:, ::-1]      
                    count = count + 1                
                                
                
        np.save(dataset_path + "/" + x_filename, X)
    else:
        X = np.load(dataset_path + "/" + x_filename)
        
    if not os.path.exists(dataset_path + "/" + y_filename):
        count = 0
        for i in range(3):
            if train==True:
                if mask_folders[i]=="channel_mask":  #Reserve the "channel" shots for testing
                    continue
            else:
                if mask_folders[i]!="channel_mask":  
                    continue
                                      
            files = next(os.walk(dataset_path + "/" + mask_folders[i]))[2]
            for file in files:
                print(str(count) + " " + file)                
                img = imread(dataset_path + "/" + mask_folders[i] + "/" + file)
                img = np.expand_dims(resize(img, (image_height, image_width), mode='constant', preserve_range=True), axis=-1)
                Y[count] = img
                count = count + 1
                
                if train==True:
                    #Flip image horizontally       
                    Y[count] = img[:, ::-1]    
                    count = count + 1   
                                
        np.save(dataset_path + "/" + y_filename, Y)
    else:
        Y = np.load(dataset_path + "/" + y_filename)
            
    return X, Y



if __name__ == '__main__':
    load_tampere()
    #load_tampere()