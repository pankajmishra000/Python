# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 12:19:21 2019

@author: Pankaj Mishra
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.datasets.mnist
from torchvision import transforms
from tqdm import tqdm
import dataset_motor_simaese as dms
import siamese_network
import time

batch_size = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Reading data
data = dms.motor_data(batch_size)

#loading the saved model
model = siamese_network.Net().to(device)
model.load_state_dict(torch.load('model_siamese_motor.pt'))
model.eval()
def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
result = []
for batch_idx, (i1, i2) in enumerate(data.test_loader):
        img1, lb1 = i1
        img2, lb2 = i2
        concatenated = torch.cat((img1.reshape(224,224), img2.reshape(224,224)),0)
        img1 = img1.to(device)
        img2 = img2.to(device)
       
#        lb1 = lb1.type(torch.LongTensor).to(device)
#        lb2 = lb2.type(torch.LongTensor).to(device)
        output1, output2 = model(img1, img2)
        euclid_distance = F.pairwise_distance(output1, output2)
        result.append([euclid_distance.item(), lb1.item(), lb2.item()])
        #print(f'Dissimilarity score is: {euclid_distance.item()}')
print(result)
plt.figure(1)
plt.subplot(121)
plt.imshow(img1.cpu().reshape(224,224))
plt.subplot(122)
plt.imshow(img2.cpu().reshape(224,224))
plt.title(f'Dissimilarity score is: {euclid_distance.item()}')
time.sleep(10)

