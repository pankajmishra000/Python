import torch.nn as nn

"""
# Encode-----------------------------------------------------------
x = Conv2D(32, (4, 4), strides=2 , activation='relu', padding='same')(input_img)
x = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
x = Conv2D(64, (4, 4), strides=2, activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
x = Conv2D(128, (4, 4), strides=2, activation='relu', padding='same')(x)
x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
encoded = Conv2D(1, (8, 8), strides=1, padding='same')(x)

# Decode---------------------------------------------------------------------
x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(encoded)
x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (4, 4), strides=2, activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), strides=1, activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (4, 4), strides=2, activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), strides=1, activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same')(x)
x = UpSampling2D((4, 4))(x)
x = Conv2D(32, (4, 4), strides=2, activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (8, 8), activation='sigmoid', padding='same')(x)
"""
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1), # image size 64
            nn.LeakyReLU(0.2,True),

            nn.Conv2d(32, 32, 4, stride=2, padding=1), # image size 32
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(32, 32, 3, stride=1, padding=1), # image size is 32
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(32, 64, 4, stride=2, padding=1), # image size is 16
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 64, 3, stride=1, padding=1), # Image size is 16
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1), # image size 8
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 64, 3, stride=1, padding=1), # Image size 8
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 100, 3, stride=1, padding=1), #Image size is 8
            nn.LeakyReLU(0.2, True),

            # nn.Conv2d(32, 100, 8, stride=1, padding=0),


        )


        self.decoder = nn.Sequential(
            # nn.Conv2d(100, 32, 8, stride=1, padding=0),
            # nn.LeakyReLU(0.2, True),

            nn.Conv2d(100, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode ='nearest'),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode ='nearest'),

            nn.Conv2d(128, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode ='nearest'),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode ='nearest'),

            nn.Conv2d(64, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode ='nearest'),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=2, mode ='nearest'),


            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Upsample(scale_factor=4, mode ='nearest'),

            nn.Conv2d(32, 1, 4, stride=2, padding=1),
            nn.Sigmoid()

        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == "__main__":
    model = autoencoder()
    print((model))
