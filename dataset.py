import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Samples are saved in (sample_count, width, height, 3) shape
class Dataset:
    # https://www.cs.toronto.edu/~kriz/cifar.html
    # All files will be in dataset/CIFAR-100/, no subfolders
    # Train data files:
    #   train
    # Val data files:
    #   test


    def rgb2gray(self, rgb):
        # transform rgb image to grayscale image
        gray_value = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        gray_img = gray_value.astype(np.uint8)
        return gray_value


    def load_cifar100(self):
        # TODO: Load dataset

        with open("./dataset/CIFAR-100/train", 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

        self.x = dict[b'data']
        self.labels = dict[b'fine_labels']

        self.x = np.dstack((self.x[:, :1024], self.x[:, 1024:2048], self.x[:, 2048:]))
        self.x = self.x.reshape((self.x.shape[0], 32, 32, 3))

        print("Shape of input rgb data:")
        print(self.x.shape)

        grayscale_list =[]

        # convert all images from train dataset to grayscale
        for img in self.x:
            grayscale_img = self.rgb2gray(img)
            grayscale_list.append(grayscale_img)

        self.x_grayscale = np.array(grayscale_list) # numpy array with grayscale images

        self.x_grayscale = self.x_grayscale.reshape(self.x.shape[0], self.x.shape[1], self.x.shape[2], 1)
        print("Shape of input converted grayscale data:")
        print(self.x_grayscale.shape)

        self.x_train = self.x_grayscale[:80,:] # inputs - grayscale images
        self.y_train = self.x[:80,:] # ground truth - rgb images

        '''
        # show some samples from train data
        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(self.x_train[i], cmap=plt.get_cmap('gray'))
        plt.show()

        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(self.y_train[i])
        plt.show()
        '''
        self.x_val = self.x_grayscale[80:,:]# inputs - grayscale images
        self.y_val = self.x[80:,:] # ground truth - rgb images
        '''
        # show some samples from val data
        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(self.x_val[i], cmap=plt.get_cmap('gray'))
        plt.show()

        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(self.y_val[i])
        plt.show()
        '''
        self.check_dataset()




    # ImageNet file structure
    # https://patrykchrabaszcz.github.io/Imagenet32/
    # All files will be in dataset/ImageNet/, no subfolders
    # Train data files:
    #   train_data_batch_1, train_data_batch_2, .., train_data_batch_10
    # Val data files:
    #   val_data
    def load_imagenet(self):
        # TODO: Load dataset
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        #self.check_dataset()


    def check_dataset(self):
        # Train data have the same resolution and number of channels as validation data
        for dim_idx in range(1,4):
            assert self.x_train.shape[dim_idx] == self.x_val.shape[dim_idx]

        # Train data has same number of samples and labels
        assert self.x_train.shape[0] == self.y_train.shape[0]

        # Validation data has same number of samples and labels
        assert self.x_val.shape[0] == self.y_val.shape[0]


    @property
    def train_data(self):
        return (self.x_train, self.y_train)

    @property
    def val_data(self):
        return (self.x_val, self.y_val)

    @property
    def train_count(self):
        return self.x_train.shape[0]

    @property
    def val_count(self):
        return self.x_val.shape[0]

    @property
    def resolution(self):
        return self.x_train.shape[0]
