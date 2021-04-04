from tensorflow import keras
from dataset import Dataset
import sys

class DeOldify(keras.Model):
    # TODO: Initialize the model
    def __init__(self):
        self.load_weights = False # Start new training by default
        self.starting_epoch = 0

    # TODO: Build the entire model
    def build_model(self):
        pass

    # TODO: Create the generator neural network
    def create_generator(self):
        pass

    # TODO: Create the discriminator neural network
    def create_discriminator(self):
        pass

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
        self.dataset = Dataset()
        if dataset_name == "ImageNet":
            self.dataset.load_imagenet()
        elif dataset_name == "CIFAR-100":
            self.dataset.load_cifar100()
        else:
            print("ERROR: Selected invalid dataset.")
            sys.exit(1)
        

    # TODO: Batch provider, implemented as a generator, returning data with shape (batch_size, width, height, 3)
    # Note: We could possibly try to add labels later, that probably would make network work better
    def batch_provider(self):
        pass
    
    # TODO: Colorize images that were selected in arguments and quit
    def colorize_selected_images(self, images):
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
