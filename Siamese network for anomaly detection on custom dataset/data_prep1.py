# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:59:59 2019

@author: Pankaj Mishra
"""

import PIL
from os import listdir
import matplotlib.pyplot as plt
from matplotlib import image
import numpy as np

def load_train_image_label():
    good_loaded_images = list()
    anomaly_loaded_images = list()
    for filenames in listdir("D:\\1st year\\Beantech data\\wetransfer-3d9539\\Preprocessed\\Default\\Conformi_prep"):
        #load image
        img_data = image.imread("D:\\1st year\\Beantech data\\wetransfer-3d9539\\Preprocessed\\Default\\Conformi_prep\\"+filenames)
        # store the loaded image
        good_loaded_images.append(img_data)
        print(f'the loaded good images are {filenames} and the shape is , {img_data.shape}')
        
    for filenames in listdir("D:\\1st year\\Beantech data\\wetransfer-3d9539\\Preprocessed\\Default\\Scarti_prep"):
        #load image
        img_data = image.imread("D:\\1st year\\Beantech data\\wetransfer-3d9539\\Preprocessed\\Default\\Scarti_prep\\"+filenames)
        # store the loaded image
        anomaly_loaded_images.append(img_data)
        print(f'the loaded anomaly images are {filenames} and the shape is , {img_data.shape}')
    # Preparing good images with the label
    labels_good_images = np.zeros(len(good_loaded_images), dtype = float)
    train_image_label = tuple(zip(good_loaded_images, labels_good_images))
    
    # Preparing bad images for test with the label
    labels_anomaly_images = np.ones(len(anomaly_loaded_images), dtype = float)
    test_image_label = tuple(zip(anomaly_loaded_images,labels_anomaly_images ))    
       
    
    return train_image_label, test_image_label
    
if __name__ == '__main__':
    train, test = load_train_image_label()
    plt.imshow(train[1][0])