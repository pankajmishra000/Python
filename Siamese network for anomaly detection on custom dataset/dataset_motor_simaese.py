# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 15:05:56 2019

@author: Pankaj Mishra
"""

import torch
from torchvision import transforms
import numpy as np
import data_prep1 
import matplotlib.pyplot as plt

class motor_data:
    def __init__(self, batch_size, norm_class = 0, nu = 0.04):
        
        T = transforms.Compose([
            #                            transforms.ToPILImage(),
            #                            transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    
        normal_image_label, anomaly_image_label = data_prep1.load_train_image_label()
#        print(train_image_label)
    
        normal_image= torch.stack([T(data) for data , label in normal_image_label])
        normal_label = torch.stack([torch.tensor(np.array(label)) for data , label in normal_image_label])
    
        anomaly_image = torch.stack([T(data) for data,label in anomaly_image_label])
        anomaly_label = torch.stack([torch.tensor(np.array(label)) for data,label in anomaly_image_label])
        print(f'length of good image {normal_image.shape}, and the toal anomaly images are {anomaly_image.shape}')
             
        # We need to creat balanced pairs
        # For our case we will create just 500 pairs.
        # nu will decide the percentage of opposite (anonmaly- normal) pairs in total of 500 pairs.
        # By default nu value is 0.04
        
        # build training set with a pair of both normal and anom data, proportion is defined by 'nu'
        perm = torch.randperm(500) # Generating only 500 samples
#        print(perm)
        perma = torch.randperm(int(nu * 500))
# generating image and labels for the normal and anomly 500 images. First set of images
        anom = anomaly_image[perma] # subsetting the anomaly images 
        normal = normal_image[perm] # subsetting the normal images       
        normal[perma] = anom # replacing the anomaly images from the anomaly images at their conrresponding randomly geenrated postiion
        print(normal.shape)       

        anom_label = anomaly_label[perma]
        norm_label = normal_label[perm]
        norm_label[perma] = anom_label
        first_Set = zip(normal,norm_label)
 
# Generating the second set of images which will be paried with the first set of images       
        perm2 = torch.randint(500,1000,(1,500))
        normal_pair =normal_image[perm2] 
        normal_pair_label = normal_label[perm2]
        second_set = zip(normal_pair.squeeze(0), normal_pair_label.squeeze(0))
        '''
        My target for the test set is to use the 500 image set pair. Where  we can use all the anomaly images for the testing and rest complementing 500 images
        as the normal image pairs.means-> 31 anomaly-normal image pair and 469 normal-normal image pairs

        '''       
# Generating test set, we will use the all anomaly images and the same amount of normal images to check how our model behaves
        perma2 = torch.randperm(anomaly_image.size(0))
        anomal_test_image = anomaly_image[perma2]
        anomal_test_label = anomaly_label[perma2]
#        first_set_test = zip(anomal_test_image, anomal_test_label)
        perm3 = torch.randint(1000,1500,(1,500-anomaly_image.size(0))) # Generate random sample of normal image - total anomaly images
        normal_test_image = normal_image[perm3]
        normal_test_label = normal_label[perm3]
        test_first_set_image = torch.cat((anomal_test_image, normal_test_image.squeeze(0)),0)
        test_first_set_lable = torch.cat((anomal_test_label, normal_test_label.squeeze(0)),0)
        test_first_Set = zip(test_first_set_image, test_first_set_lable)
        
        perm4 = torch.randint(0,1000,(1,500)) # resample any normal image between 0, 1000 images. Only 500 hundered random samples will be generated
        
        normal_test_image_pair = normal_image[perm4]
        normal_test_label_pair = normal_label[perm4]
        test_pair_set = zip(normal_test_image_pair.squeeze(0), normal_test_label_pair.squeeze(0))        
         
        
        
######### Final train and test set############
        train = tuple(zip(first_Set, second_set))        
        test = tuple(zip(test_first_Set, test_pair_set))
        
########### train and test loader ###########
        self.train_loader  = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
if __name__ == '__main__':
    d = motor_data(32)
    
    for i1, i2 in d.train_loader:
        print(len(i1))
        print(i1[0].shape)
        plt.imshow(i1[0][0])
        break
    