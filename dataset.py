import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob
import random

def convert_all_imgs_to_grayscale(x):
    # convert all images from dataset to grayscale
    grayscale_list = np.array([], dtype=np.uint8)

    for img in x:
        grayscale_img = rgb2gray(img)
        grayscale_list = np.append(grayscale_list, grayscale_img)

    return grayscale_list.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

def rgb2gray(rgb):
    # transform rgb image to grayscale image
    gray_value = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    gray_img = gray_value.astype(np.uint8)
    return gray_img

# Samples are saved in (sample_count, width, height, 3) shape
class Dataset:
    def __init__(self, resolution=64):
        self.res = resolution
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None

    # Note: We could possibly try to add labels later, that probably would make network work better
    def batch_provider(self, batch_size, train=True, convert_range=True):
        while True:
            if train:
                indices = np.random.randint(0, len(self.x_train), batch_size)
                batch_samples = np.array(self.x_train[indices], dtype=np.uint8)
                batch_targets = np.array(self.y_train[indices], dtype=np.uint8)
                grayscale_images = convert_all_imgs_to_grayscale(self.x_train[indices])
            else:
                indices = np.random.randint(0, len(self.x_val), batch_size)
                batch_samples = np.array(self.x_val[indices], dtype=np.uint8)
                batch_targets = np.array(self.y_val[indices], dtype=np.uint8)
                grayscale_images = convert_all_imgs_to_grayscale(self.x_val[indices])
                        
            # Convert images to <-1;1> range from <0;255> range
            if convert_range:
                batch_samples = (batch_samples - 127.5) / 127.5
                grayscale_images = (grayscale_images - 127.5) / 127.5

            yield batch_samples, batch_targets, grayscale_images

    def get_input_img(self, x):
        x = np.dstack((x[:, :self.res**2], x[:, self.res**2:2*(self.res**2)], x[:, 2*self.res**2:]))
        x = x.reshape((x.shape[0], self.res, self.res, 3))
        return x

    def load_imagenet(self):
        # process train data
        self.y_train = np.array([])

        i = 0
        # cycle for concatenate all batches from image net together
        for file in glob.glob("./dataset/ImageNet/*"):
            file_substr = file.split('/')[-1] # get name of processed file
            if (file_substr == "val_data"):
                continue
            with open(file, 'rb') as fo:
                train_batch_dict = pickle.load(fo, encoding='bytes')

            x_batch_train = train_batch_dict['data']
            x_batch_train = self.get_input_img(x_batch_train)
            y_batch_train = train_batch_dict['labels']
            self.y_train = np.concatenate((self.y_train, y_batch_train))

            if (i == 0):
                self.x_train = x_batch_train
            else:
                self.x_train = np.append(self.x_train, x_batch_train)

            i = i + 1
            
        print("Shape of train data:")
        print(self.x_train.shape)

        # process val data
        with open("./dataset/ImageNet/val_data", 'rb') as fo:
            val_dict = pickle.load(fo, encoding='bytes')

        self.x_val = val_dict['data']
        y_val_list = val_dict['labels']
        self.y_val = np.array(y_val_list)

        self.x_val = self.get_input_img(self.x_val)
        print("Shape of val data:")
        print(self.x_val.shape)

        self.check_dataset()

    def preview_data(self):
        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(self.x_train[random.randint(0, len(self.x_train))])
        plt.show()

        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(self.x_val[random.randint(0, len(self.x_val))])
        plt.show()

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
        return self.x_train, self.y_train

    @property
    def val_data(self):
        return self.x_val, self.y_val

    @property
    def train_count(self):
        return self.x_train.shape[0]

    @property
    def val_count(self):
        return self.x_val.shape[0]
