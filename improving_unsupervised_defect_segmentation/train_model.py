import os
import numpy as np

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import load_texture_data
from autoencoder_conv import autoencoder

import torch.utils.data

if not os.path.exists('./training'):
    os.mkdir('./training')

def to_img(x):
    x = x.view(x.size(0),1,128,128)
    #print (x)
    return x


num_epochs = 1000
batch_size = 128
learning_rate = 2e-4

# Defining the image transforms
transform = transforms.Compose([
                                #transforms.Grayscale(1),
                                transforms.ToPILImage(),
                                transforms.Resize(256),
                                transforms.RandomCrop(128),
                                transforms.ToTensor()
                               ])


train_loader, test_loader = load_texture_data.load_train_test_images()

train_loader = torch.stack([transform(data.reshape(512,512,1)) for idx, data in enumerate(train_loader)])
test_loader = torch.stack([transform(data.reshape(512,512,1)) for idx, data in enumerate(test_loader)])

train_data_loader = torch.utils.data.DataLoader(train_loader, batch_size = batch_size, shuffle = True)
test_data_loader = torch.utils.data.DataLoader(test_loader, batch_size = batch_size, shuffle = False)

model = autoencoder()
model = model.cuda()
print(model)

#
reconstruction_function = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1e-6)


# Module: pytorch_ssim.SSIM(window_size = 11, size_average = True)
# ssim_loss = pytorch_ssim.SSIM()

import matplotlib.pyplot as plt

#plt.plot(train_loader[0])
#plt.show()

def train(epoch):
    model.train()
    train_loss = 0
    loss_list = []

    for batch_idx, data in enumerate(train_data_loader):
        img = data
        #img = img.view(img.size(0), -1)
        img = Variable(img)
        img = img.cuda()


        optimizer.zero_grad()

        # track history
        with torch.set_grad_enabled(True):
            recon_batch = model(img)



            #loss = loss_2
            loss = reconstruction_function(recon_batch, img)

            loss.backward()

            # for printing the loss
            #loss_avg = reconstruction_function(recon_batch, img)
            train_loss += loss

            optimizer.step()

            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(img),
                    len(train_data_loader), 100. * batch_idx / len(train_data_loader),
                    loss.item() / len(img)))

    #print('====> Epoch: {} Average loss: {:.6f}'.format(
    #    epoch, train_loss / len(train_loader)))

    if epoch % 10 == 0:
        x = to_img(img.cpu().data)
        x_hat = to_img(recon_batch.cpu().data)
        #save_image(x, './training/x_{}.png'.format(epoch))
        #save_image(x_hat, './training/x_hat_{}.png'.format(epoch))

    torch.save(loss_list, './train_loss.pt')

#if __name__ == "__main__":

# TRAINING THE MODEL
print("Training for %d epochs..." % num_epochs)
for epoch in range(num_epochs):
    train(epoch)

torch.save(model.state_dict(),'./autoencoder_model.pth')