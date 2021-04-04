import pickle
import numpy as np
import matplotlib.pyplot as plt
import glob

# Samples are saved in (sample_count, width, height, 3) shape
class Dataset:
    # https://www.cs.toronto.edu/~kriz/cifar.html
    # All files will be in dataset/CIFAR-100/, no subfolders
    # Train data files:
    #   train
    # Val data files:
    #   test

    # x - input grayscale image
    # y - label of number
    # gt - ground truth - original rgb image


    def rgb2gray(self, rgb):
        # transform rgb image to grayscale image
        gray_value = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
        gray_img = gray_value.astype(np.uint8)
        return gray_value

    def get_input_img32(self, x):
        x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
        x = x.reshape((x.shape[0], 32, 32, 3))
        return x

    def convert_all_imgs_to_grayscale(self, x):
        # convert all images from dataset to grayscale
        grayscale_list =[]

        for img in x:
            grayscale_img = self.rgb2gray(img)
            grayscale_list.append(grayscale_img)

        x_grayscale = np.array(grayscale_list) # numpy array with grayscale images
        return x_grayscale


    def load_cifar100(self):
        # TODO: Load dataset

        with open("./dataset/CIFAR-100/train", 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')

        self.x = dict[b'data']
        y_list = dict[b'fine_labels']
        self.y = np.array(y_list)
        print(self.y)

        self.x = self.get_input_img32(self.x)

        print("Shape of input rgb data:")
        print(self.x.shape)

        self.x_grayscale = self.convert_all_imgs_to_grayscale(self.x) # numpy array with grayscale images

        self.x_grayscale = self.x_grayscale.reshape(self.x.shape[0], self.x.shape[1], self.x.shape[2], 1)
        print("Shape of input converted grayscale data:")
        print(self.x_grayscale.shape)

        self.x_train = self.x_grayscale[:40000,:] # inputs - grayscale images
        #print(self.x_train.shape)
        self.y_train = self.y[:40000] # labels
        #print(self.y_train.shape)
        self.gt_train = self.x[:40000,:] # ground truth - rgb images
        #print(self.gt_train.shape)

        '''
        # show some samples from train data
        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(self.x_train[i], cmap=plt.get_cmap('gray'))
        plt.show()

        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(self.gt_train[i])
        plt.show()
        '''


        self.x_val = self.x_grayscale[40000:,:]# inputs - grayscale images
        self.y_val = self.y[40000:] # labels
        self.gt_val = self.x[40000:,:] # ground truth - rgb images

        '''
        # show some samples from val data
        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(self.x_val[i], cmap=plt.get_cmap('gray'))
        plt.show()

        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(self.gt_val[i])
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
        # process train data
        y_train_list = []

        i = 0
        # cycle for concatenate all batches from image net together
        for file in glob.glob("./dataset/ImageNet/*"):
            file_substr = file.split('/')[-1] # get name of processed file
            if (file_substr == "val_data"):
                continue
            with open(file, 'rb') as fo:
                train_batch_dict = pickle.load(fo, encoding='bytes')

            x_batch_train = train_batch_dict['data']
            x_batch_train = self.get_input_img32(x_batch_train)
            #print(x_batch_train.shape)
            y_batch_train = train_batch_dict['labels']
            y_train_list = y_train_list + y_batch_train
            #print(y_batch_train)

            if (i == 0):
                x_train_np = x_batch_train
            else:
                print(x_train_np.shape)
                x_train_np = np.concatenate((x_train_np, x_batch_train))
                print(x_train_np.shape)

            i = i + 1

        self.gt_train = x_train_np # ground truth - rgb images
        print("Shape of input train rgb data:")
        print(self.gt_train.shape)
        self.y_train = np.array(y_train_list) # labels

        # convert all ground truths to grayscale
        self.x_train = self.convert_all_imgs_to_grayscale(self.gt_train)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], self.x_train.shape[2], 1)
        print("Shape of input converted train grayscale data:")
        print(self.x_train.shape)

        '''
        # show some samples from train data
        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(self.x_train[i], cmap=plt.get_cmap('gray'))
        plt.show()

        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(self.gt_train[i])
        plt.show()
        '''

        # process val data
        with open("./dataset/ImageNet/val_data", 'rb') as fo:
            val_dict = pickle.load(fo, encoding='bytes')

        self.x = val_dict['data']
        y_val_list = val_dict['labels']
        self.y_val = np.array(y_train_list)

        self.gt_val = self.get_input_img32(self.x)
        print("Shape of input val rgb data:")
        print(self.gt_val.shape)

        self.x_val = self.convert_all_imgs_to_grayscale(self.gt_val)
        self.x_val = self.x_val.reshape(self.x_val.shape[0], self.x_val.shape[1], self.x_val.shape[2], 1)
        print("Shape of input converted val grayscale data:")
        print(self.x_val.shape)

        '''
        # show some samples from val data
        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(self.x_val[i], cmap=plt.get_cmap('gray'))
        plt.show()

        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(self.gt_val[i])
        plt.show()
        '''

        self.check_dataset()


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
