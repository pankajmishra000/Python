# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:53:02 2019

@author: Pankaj Mishra
"""

import torch
from torchvision import transforms
import numpy as np
import data_prep1 

class motor_data:
    def __init__(self, batch_size, norm_class = 0, nu = 0.00):
        
        T = transforms.Compose([
            #                            transforms.ToPILImage(),
            #                            transforms.RandomCrop(32, padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
    
        train_image_label, test_image_label = data_prep1.load_train_image_label()
#        print(train_image_label)
    
        train_image= [data for data , label in train_image_label]
        train_label = [label for data , label in train_image_label]
    
        test_image = [data for data,label in test_image_label]
        test_label = [label for data,label in test_image_label]
       # print(train_image[0].shape)
        
    
    
        train_normal = torch.stack( [T(train_image[key].reshape(224,224,1)) for (key, label) in enumerate(train_label) if label == norm_class] )
        print('train_normal:', train_normal.shape)
        #    train_anom   = torch.stack( [T(train_image[key]) for (key, label) in enumerate(train_label) if label!= norm_class] )
        #    test_normal  = torch.stack( [T(test_image[key]) for (key, label) in enumerate(test_label)  if label == norm_class] )
        #print(train_normal[0])
        test_anom    = torch.stack( [T(test_image[key].reshape(224,224,1)) for (key, label) in enumerate(test_label)  if label!= norm_class] )
        train_label_normal = torch.stack( [torch.tensor(np.array(train_label[key])) for (key, label) in enumerate(train_label) if label == norm_class] )
        #    train_label_anom   = torch.stack( [torch.from_numpy(np.array(train_label[key])) for (key, label) in enumerate(train_label) if label != norm_class] )
        #    test_label_normal  = torch.stack( [torch.from_numpy(np.array(test_label[key]))  for (key, label) in enumerate(test_label)  if label== norm_class] )
        test_label_anom    = torch.stack( [torch.tensor(np.array(test_label[key]))  for (key, label) in enumerate(test_label)  if label!= norm_class] )

    # build training set with a mix of normal and anom data, proportion is defined by 'nu'
#    perm = torch.randperm(train_anom.size(0))
    #        print(perm)
#    samples = perm[:int(nu * train_normal.size(0))]
    #        print(samples)
#    train_anom = train_anom[ samples ]
    #        print('train anom:',train_anom)
#    train_label_anom = train_label_anom [ samples ]
    
        train = tuple(zip(train_normal, train_label_normal))
        test = tuple(zip(test_anom, test_label_anom))
   
    
        self.train_loader  = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)

        #        print('after train loader ',self.train_loader.batch_sampler)
        self.test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    
def plot_images_separately(images):
    import matplotlib.pyplot as plt
    import matplotlib
    fig = plt.figure()
    for j in range(1, images.size(0)//4+1):
        ax = fig.add_subplot(2, images.size(0)//8+1, j)
        ax.matshow(images[j-1], cmap = matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.show()
    plt.pause(0.01)

if __name__ == "__main__":
    m = motor_data(1)
    lab = []
    for i, data in enumerate(m.train_loader):
        #        print(i)
        #         print(data[0])
        # plot_images_separately(data[0][:,0])
        img, label = data
        # img = img/255.
#        print(img[10])
#        print((img[10]).max())
#        print((img[10]).min())
        plot_images_separately(img[:,0])
        print("batch size: {}, label: {}".format(img.size(), label))
        lab.append(label)
#        break
