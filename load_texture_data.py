
#Loading Data
from os import listdir
import matplotlib.pyplot as plt
from matplotlib import image

def load_train_test_images():
    train_good_loaded_images = list()
    test_anom_loaded_images = list()
    for train_img in listdir("D:/1st year/Python codes/Improving unsupervised defect segmentation/textures/texture_1/train/good"):
           train_image_data = image.imread("D:/1st year/Python codes/Improving unsupervised defect segmentation/textures/texture_1/train/good/"+train_img)
           train_good_loaded_images.append(train_image_data)
    print(f'total good train image loaded{len(train_good_loaded_images)} shape of image {train_image_data.shape}')

    for train_img in listdir("D:/1st year/Python codes/Improving unsupervised defect segmentation/textures/texture_1/test/defective"):
           train_image_data = image.imread("D:/1st year/Python codes/Improving unsupervised defect segmentation/textures/texture_1/test/defective/"+train_img)
           test_anom_loaded_images.append(train_image_data)
    print(f'total good train image loaded{len(test_anom_loaded_images)} shape of image {train_image_data.shape}')

    return train_good_loaded_images, test_anom_loaded_images




if __name__ == "__main__":
    good, anomaly = load_train_test_images()
    print(good[0].max())
    print(anomaly[0].max())

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(good[0])
    plt.subplot(122)
    plt.imshow(anomaly[0])
    plt.show()


