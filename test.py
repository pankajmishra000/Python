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
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def to_img(x):
    x = x.view(x.size(0),1,128,128)
    #print (x)
    return x


def test():
    model.load_state_dict(torch.load('./autoencoder_model.pth'))
    model.eval()
    count = 0
    loss_list = []
    img_orig = []
    recon_x = []
    train_loss = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(test_data_loader):
            img = data

            img = Variable(img, requires_grad=False) #.cuda()

            recon_batch = model(img)

            loss = reconstruction_function(recon_batch, img)
            img_orig.append(img.cpu().numpy())
            recon_x.append(recon_batch.cpu().numpy())

            loss_list.append(loss.item())
            train_loss += loss

            print('Loss/Train avg loss: {} / {}'.format(loss, train_loss))

            x = to_img(img.cpu().data)
            x_hat = to_img(recon_batch.cpu().data)
            #save_image(x, './test_vae/x_{}.png'.format(count))
            #save_image(x_hat, './test_vae/x_hat_{}.png'.format(count))
            count += 1

        #torch.save(loss_list, './train_loader_loss.pt')
        #torch.save(target_t, './test_target_t.pt')
        torch.save(img_orig, './img_orig.pt')
        torch.save(recon_x, './recon_x.pt')


def show():
    decoded_imgs = torch.load('./recon_x.pt')
    x_train = torch.load('./img_orig.pt')

    # for batch_idx, data in enumerate(decoded_imgs):
    #     x_train.append(data)

    n = 8 #how many digits we will display
    plt.figure(figsize=(20, 5), dpi=100)
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_train[i].reshape(128, 128))
        plt.gray()
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(False)

        # SSIM Encode
        ax.set_title("Encode_Image")

        npImg = x_train[i]
        npImg = npImg.reshape((128, 128))
        formatted = (npImg * 255 / np.max(npImg)).astype('uint8')
        img = Image.fromarray(formatted)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(128, 128))
        plt.gray()
        ax.get_xaxis().set_visible(True)
        ax.get_yaxis().set_visible(False)

        # SSIM Decoded
        npDecoded = decoded_imgs[i]
        npDecoded = npDecoded.reshape((128, 128))
        formatted2 = (npDecoded * 255 / np.max(npDecoded)).astype('uint8')
        decoded = Image.fromarray(formatted2)

        value = ssim(img, decoded)

        label = 'SSIM: {:.3f}'

        ax.set_title("Decoded_Image")
        ax.set_xlabel(label.format(value))

    plt.show()


if __name__ == "__main__":
    batch_size = 1

    reconstruction_function = nn.BCELoss()
    criterion2 = nn.BCELoss()

    # Defining the image transforms
    transform = transforms.Compose([
        # transforms.Grayscale(1),
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.RandomCrop(128),
        transforms.ToTensor()
    ])

    train_loader, test_loader = load_texture_data.load_train_test_images()


    train_loader = torch.stack([transform(data.reshape(512, 512, 1)) for idx, data in enumerate(train_loader)])
    test_loader = torch.stack([transform(data.reshape(512, 512, 1)) for idx, data in enumerate(test_loader)])

    train_data_loader = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle=False)


    model = autoencoder()
    # model = model.cuda()

    print("Testing...")

    test()
    show()
