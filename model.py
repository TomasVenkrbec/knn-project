from tensorflow.keras import Model
from tensorflow.keras.layers import Input, AveragePooling2D, UpSampling2D, Concatenate, Flatten, Dense
from tensorflow.keras.utils import plot_model
from SpectralNormalization import ConvSN2D
from SelfAttentionLayer import SelfAttention
from dataset import Dataset

import sys

GENERATOR_MIN_RESOLUTION = 8 # Resolution in "deepest" layer of generator U-net

class DeOldify(Model):
    def __init__(self, resolution=64, filters_gen=32, filters_disc=16):
        super().__init__()
        self.load_weights = False # Start new training by default
        self.starting_epoch = 0
        self.resolution = resolution # Resolution of train images
        self.filters_gen = filters_gen # Number of convolutional filters in first and last layer of generator
        self.filters_disc = filters_disc # Number of convolutional filters in first layer of discriminator

    # TODO: Build the entire model
    def build_model(self):
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()

    def create_generator(self):
        layer_outputs = [] # Outputs of last convolution in the "left side" of U-net, which will be copied to "right side" of U-net

        grayscale_img = Input(shape=(self.resolution, self.resolution, 1)) # Grayscale image
        gen = ConvSN2D(self.filters_gen, kernel_size=3, padding="same")(grayscale_img)
        gen = SelfAttention(self.filters_gen)(gen)
        gen = ConvSN2D(self.filters_gen, kernel_size=3, padding="same")(gen)
        gen = SelfAttention(self.filters_gen)(gen)
        layer_outputs.append(gen) # Will be connected to output layer later

        filters = self.filters_gen * 2
        resolution = self.resolution // 2

        # Create all layers in "left side" of U-net
        while resolution >= GENERATOR_MIN_RESOLUTION:
            gen = AveragePooling2D()(gen)
            gen = ConvSN2D(filters, kernel_size=3, padding="same")(gen)
            gen = SelfAttention(filters)(gen)
            gen = ConvSN2D(filters, kernel_size=3, padding="same")(gen)
            gen = SelfAttention(filters)(gen)
            
            # Take the output of this layer so it can be connected as input to "right side" of U-net
            layer_outputs.append(gen)

            # Next layer has half the resolution and twice the filters
            resolution //= 2
            filters *= 2

        # Decrease number of filters twice (one calculation in cycle was excessive)
        filters //= 4

        # Create all layers in "right side" of U-net
        for left_side_out in reversed(layer_outputs[:-1]): # Skip the last layer output (we upscaled that already) and take layers in reverse
            # Upscale the output of previous layer and halve the filter count
            gen = UpSampling2D()(gen)
            gen = ConvSN2D(filters, kernel_size=2, padding="same")(gen)
            gen = SelfAttention(filters)(gen)
            
            # Concatenate the upscaled previous layer output with "left tree" output with the same resolution
            concat = Concatenate()([left_side_out, gen])

            # Convolution block
            gen = ConvSN2D(filters, kernel_size=3, padding="same")(concat)
            gen = SelfAttention(filters)(gen)
            gen = ConvSN2D(filters, kernel_size=3, padding="same")(gen)
            gen = SelfAttention(filters)(gen)

            # Next layer has twice the resolution and half the filters
            filters //= 2
        
        # Change the number of feature maps to 3, so we get final RGB image
        gen = ConvSN2D(3, kernel_size=1, activation="tanh", padding="same")(gen)
        output_gen = SelfAttention(3)(gen)

        # Create the generator model
        generator_model = Model(inputs=grayscale_img, outputs=output_gen)
        generator_model.summary()
        plot_model(generator_model, to_file="gen_model.png", show_shapes=True, show_dtype=True, show_layer_names=True)
        return generator_model
            
    def create_discriminator(self):
        rgb_img = Input(shape=(self.resolution, self.resolution, 3)) # RGB image

        # Convolutional block with attention
        disc = ConvSN2D(self.filters_disc, strides=2, kernel_size=3, padding="same")(rgb_img)
        disc = SelfAttention(self.filters_disc)(disc)
        disc = ConvSN2D(self.filters_disc, kernel_size=3, padding="same")(disc)
        disc = SelfAttention(self.filters_disc)(disc)

        resolution = self.resolution // 2
        filters = self.filters_disc * 2
        while resolution > 4:
            disc = ConvSN2D(filters, strides=2, kernel_size=3, padding="same")(disc)
            disc = SelfAttention(filters)(disc)
            disc = ConvSN2D(filters, kernel_size=3, padding="same")(disc)
            disc = SelfAttention(filters)(disc)
            
            resolution //= 2 # Halve the resolution
            filters *= 2 # Twice the filters
        
        # Output block
        disc = ConvSN2D(filters, kernel_size=4, padding="valid")(disc)
        disc = SelfAttention(filters)(disc)
        disc = Flatten()(disc)
        prob = Dense(1, activation='sigmoid')(disc)

        # Build the discriminator model
        discriminator_model = Model(inputs=rgb_img, outputs=prob)
        discriminator_model.summary()
        plot_model(discriminator_model, to_file="disc_model.png", show_shapes=True, show_dtype=True, show_layer_names=True)
        return discriminator_model

    # TODO: Generate various graphs
    # - generator loss
    # - discriminator loss and accuracy (if we use BCE)
    # - batch of generated images (maybe compared to black & white and ground truth?)
    #
    # Note: If we somehow manage to run Tensorboard (which would be great!), there won't be a reason to have this function
    def plot_graphs(self):
        pass

    # TODO: Functions for individual graphs which will be created periodically during training
    def plot_loss_graph(self):
        pass
    def plot_accuracy_graph(self):
        pass
    def plot_results(self):
        pass

    # TODO: Load dataset - ImageNet (http://image-net.org/)
    def load_dataset(self, dataset_name):
        # Inspiration: 
        # ImageNet - https://patrykchrabaszcz.github.io/Imagenet32/
        # CIFAR-100 - https://www.cs.toronto.edu/~kriz/cifar.html
        self.dataset = Dataset(self.resolution)
        if dataset_name == "ImageNet":
            self.dataset.load_imagenet()
        elif dataset_name == "CIFAR-100":
            self.dataset.load_cifar100()
        else:
            print("ERROR: Selected invalid dataset.")
            sys.exit(1)
    
    # TODO: Colorize images that were placed in 'images_to_colorize' folder and quit, colorized images are placed into 'results' folder
    def colorize_selected_images(self):
        pass
    
    # Initialize variables regarding training continuation 
    def prepare_snapshot_load(self, starting_epoch, weights_path):
        self.starting_epoch = starting_epoch
        self.weights_path = weights_path
        self.load_weights = True

    # TODO: Train the networks
    def train(self):
        # Try to use Tensorboard
        # https://www.tensorflow.org/tensorboard/get_started

        # Try to use fit() instead of manually training batch by batch
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        pass

    # TODO: Overwrite the default class train_step function, so we can train GAN with fit()
    def train_step(self, data):
        # Inspiration: https://keras.io/examples/generative/dcgan_overriding_train_step/
        pass
