import os
import numpy as np
import pandas as pd

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import torch.nn.functional as F
from mnist import Mnist

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#if not os.path.exists('./test_vae'):
 #   os.mkdir('./test_vae')


def test():
    
    count = 0
    train_loss_list = torch.load('train_loader_loss.pt')

    test_loss_list = torch.load('test_loader_loss.pt')
    test_anom_loss_list = torch.load('test_anom_loader_loss.pt')

    target_t = torch.load('test_target_t.pt')

    loss = []

    for e in test_loss_list:
        loss.append(e)

    for e in test_anom_loss_list:
        loss.append(e)


    train_loss_list = np.array(train_loss_list, dtype=np.float32)
    #test_loss_list = np.array(test_loss_list, dtype=np.float32)

    loss = np.array(loss, dtype=np.float32)

    logmodel = LogisticRegression(solver='liblinear', class_weight='balanced').fit(train_loss_list.reshape(-1,1), target_t)


    predictions = logmodel.predict(loss.reshape(-1, 1))

    y_target = []

    for i in range(len(test_loss_list)):
        y_target.append(1.)

    for i in range(len(test_anom_loss_list)):
        y_target.append(0.)

    cm = confusion_matrix(y_target, predictions)
    acc = accuracy_score(y_target, predictions)*100.
    #acc_0 = accuracy_score(y_target_0, predictions_0) * 100.

    print('Test Accuracy: {} %, \nconfusion_matrix: \n{}'.format(acc, cm))

#if __name__ == "__main__":
print("TEST ACCURACY")
test()