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
from autoencoder_conv import autoencoder
from SSIM_PIL import compare_ssim as ssim
import load_texture_data

#if not os.path.exists('./test_vae'):
 #   os.mkdir('./test_vae')


def to_img(x):
    x = x.view(x.size(0),1,128,128)
    #print (x)
    return x

batch_size = 10

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


def test_true():
    #model.eval()
    count = 0
    loss_list = []
    test_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            img = data

            img = Variable(img, requires_grad=False).cuda()

            recon_batch = model(img)

            loss = ssim(recon_batch, img)

            #loss = loss_2 + loss_1

            print(loss)
            loss_list.append(loss.item())

            test_loss += loss

            print('Loss/Test avg loss: {} / {}'.format(loss, test_loss))

            x = to_img(img.cpu().data)
            x_hat = to_img(recon_batch.cpu().data)
            #save_image(x, './test_vae/x_{}.png'.format(count))
            #save_image(x_hat, './test_vae/x_hat_{}.png'.format(count))
            count += 1


        torch.save(loss_list, './test_loader_loss.pt')


def test_anom():

    #model.eval()
    count = 0
    loss_list = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(m.test_anom_loader):
            if(count < len(m.test_loader)):
                img = data
                #img = img.view(img.size(0), -1)
                img = Variable(img, requires_grad=False).cuda()

                recon_batch = model(img)

                recon_x = Variable(recon_batch)

                #loss_1 = reconstruction_function(recon_batch, img)
                loss = -ssim_loss(recon_x, img)

                #loss = loss_2 + loss_1

                print(loss)
                loss_list.append(loss.item())
                """
                print('Loss/Train loss: {} / {}'.format(loss, train_loss))
                if (loss <= train_loss):
                    print('Classe 1')
                else:
                    print('Classe 0')
                """

                x = to_img(img.cpu().data)
                x_hat = to_img(recon_batch.cpu().data)
                # save_image(x, './test_vae/x_{}.png'.format(count))
                # save_image(x_hat, './test_vae/x_hat_{}.png'.format(count))
                count += 1

        torch.save(loss_list, './test_anom_loader_loss.pt')
    print(str(count))

model.load_state_dict(torch.load('./autoencoder_model.pth'))
#if __name__ == "__main__":
print("Testing...")
test_true()
test_anom()


