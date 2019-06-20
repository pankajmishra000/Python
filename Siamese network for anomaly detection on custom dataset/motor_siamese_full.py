# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:00:41 2019

@author: Pankaj Mishra
"""
'''
The below has two codes embeeded in one. you can run the siamese netwrok with binary cross entropy and contrastive loss both.
You just need to uncomment line 122 and 123 if you want to run with BCE loss else, it will by default run with contractive loss.
'''

# Importing libraries
import os
from PIL import Image
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms # you can perform transforms on your dataset. For my usecase dataset, I didn't use any.
import dataset_motor_simaese as dms

 # Defining the hyperparameters
do_learn = True
save_frequency = 2
batch_size = 4
lr = 0.001
num_epochs = 20
weight_decay = 0.0001

# Reading data
data = dms.motor_data(batch_size) # Change this part as per your dataset. I am not sharing the used dataset for my study due to privacy issues.

#### Siamese Net ###

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),


        )

        self.fc1 = nn.Sequential(
            nn.Linear(401408, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# ----- Loss Function -------#

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive

 #----- Defining Train------#
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = ContrastiveLoss()
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (i1, i2) in enumerate(data.train_loader):
        img1, lb1 = i1
        img2, lb2 = i2
        img1 = img1.to(device)
        img2 = img2.to(device)
        lb1 = lb1.type(torch.LongTensor).to(device)
        lb2 = lb2.type(torch.LongTensor).to(device)
        label = (lb1+lb2).type(torch.float)
        optimizer.zero_grad()
        output_positive, output_negative = model(img1, img2)
        
        
#        loss_positive = F.cross_entropy(output_positive, lb1) # uncomment this if you want to run with bnary cross entropy
#        loss_negative = F.cross_entropy(output_negative, lb2) # uncomment this if you want to run with bnary cross entropy
                  
        loss = criterion(output_positive, output_negative,label)
        loss.backward()
          
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx*batch_size, len(data.train_loader), 100. * batch_idx*batch_size / len(data.train_loader),
                loss.item()))
    
#Saving trained network
torch.save(model.state_dict(), 'model_siamese_motor.pt')

