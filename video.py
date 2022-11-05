#!/usr/bin/env python
"""
Author: Codegression
This module is to generate videes with highlighted water based on raw image sequences or input video
"""


import numpy as np
import inference
import os
import imageio
import PIL
import ffmpeg

def process_imagesequence(path, alpha=150):
    """_
    Performs inference on a folder of folders containing images and saves the output as a video in the same folder
    
    Parameters
    ----------
    path : str
        path of video file
    alpha : int
        water pixel transparency (0 to 255)

    Returns
    -------
    None.
    """
    
    folders = next(os.walk(path))[1]  
    for folder in folders:
        writer = imageio.get_writer(path + '/' + folder + '/video.avi', 
                                    format='AVI',
                                    fps=3)
    
        files = next(os.walk(path + "/" + folder))[2]   
        for file in files:      
            if os.path.splitext(file)[1].lower()!='.png' and os.path.splitext(file)[1].lower()!='.jpg' and os.path.splitext(file)[1].lower()!='.jpg' and os.path.splitext(file)[1].lower()!='.gif':
                    continue
            print(folder + "/" + file)                
           
            image = PIL.Image.open(path + "/" + folder + "/" + file)    
            outputimage, timeelapsed = inference.infer(image=image, 
                                                       alpha=alpha, 
                                                       interpolate=True)
            
            modified_img = np.asarray(outputimage)
            modified_img = imageio.core.util.Array(modified_img)
            writer.append_data(modified_img)
        
        writer.close()
        print('Saved')
    
def process_video(path, alpha=150):
    """
    This function reads a video file frame by frame and for each frame, it performs inferene and highlights water pixels.

    Parameters
    ----------
    path : str
        path of video file
    alpha : int
        water pixel transparency (0 to 255)

    Returns
    -------
    None
    """
    directory, filename = os.path.split(path)
    
    probe = ffmpeg.probe(path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    fps = int(video_info['r_frame_rate'].split('/')[0])
    
    if not os.path.exists(directory + "/output"):
        os.mkdir(directory + "/output")
    
    reader = imageio.get_reader(path)
    writer = imageio.get_writer(directory + '/output/' + filename,                               
                                fps=fps)
    for i, frame in enumerate(reader):
        img = np.asarray(frame)
        img = PIL.Image.fromarray(img)
        outputimage, timeelapsed = inference.infer(image=img, 
                                                   alpha=alpha, 
                                                   interpolate=True)
        modified_img = np.asarray(outputimage)
        modified_img = imageio.core.util.Array(modified_img)
        writer.append_data(modified_img)
        print(i)
        
    reader.close()
    writer.close()
    
    
def process_all_videos(path, alpha=150):
    """
    This function  calls process_video for all the video files in a specified folder

    Parameters
    ----------
    path : str
        folder which contains video files
    alpha : int
        water pixel transparency (0 to 255)

    Returns
    -------
    None
    """
    files = next(os.walk(path))[2]   
    for file in files:      
        if os.path.splitext(file)[1].lower()!='.avi' and os.path.splitext(file)[1].lower()!='.mp4' and os.path.splitext(file)[1].lower()!='.mov':
                continue
        
        if os.path.exists(path + "/output/" + file): #File already processed. Just skip it. Sometimes we want to resume processing half-way
            print("Skipping", file)           
            continue
        
        print()
        print("Reading", file)
        process_video(path + "/" + file, alpha)
        
        
if __name__ == '__main__':
    #process_imagesequence('../datasets/image sequence')
    process_all_videos("../datasets/video")