import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, AveragePooling2D, UpSampling2D, Concatenate, Flatten, Dense, LeakyReLU
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Mean, Accuracy
from tensorflow.keras.callbacks import TensorBoard
from SpectralNormalization import ConvSN2D
from SelfAttentionLayer import SelfAttention
from callbacks import ResultsGenerator
from dataset import Dataset, convert_all_imgs_to_grayscale, rgb2gray
from tensorflow import GradientTape, math, summary
from datetime import datetime
from utils import plot_to_image, image_grid, perceptual_loss
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import sys
import os 
from PIL import Image
import glob
import matplotlib.pyplot as plt

GENERATOR_MIN_RESOLUTION = 8 # Resolution in "deepest" layer of generator U-net
EXAMPLE_COUNT = 25 # Number of example images in Tensorboard

class DeOldify(Model):
    def __init__(self, 
                resolution=64, 
                filters_gen=32, 
                filters_disc=16, 
                generator_lr=0.0001, 
                discriminator_lr=0.0004,
                batch_size=8,
                attention_res=32,
                val_batches=100,
                epochs=100,
                output_frequency=50,
                output_count=36,
                logdir="logs",
                beta_1=0, 
                beta_2=0.9):
        super().__init__()

        # Training-related settings
        self.load_weights = False # Start new training by default
        self.weights_path = None
        self.starting_epoch = 0
        self.batch_size = batch_size
        self.attention_res = attention_res
        self.val_batches = val_batches
        self.epochs = epochs
        self.output_frequency = output_frequency
        self.output_count = output_count
        self.logdir = logdir + "/" + datetime.now().strftime("%Y%m%d-%H%M%S") 

        # Network-related settings
        self.resolution = resolution # Resolution of train images
        self.filters_gen = filters_gen # Number of convolutional filters in first and last layer of generator
        self.filters_disc = filters_disc # Number of convolutional filters in first layer of discriminator
        self.loss = BinaryCrossentropy()
        self.loss_gen = perceptual_loss

        # Hyperparameters
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2

        # Prepare labels for training (they are the same for all training steps), so we don't have to generate them every time
        self.labels_real = np.zeros((self.batch_size, 1))    
        self.labels_fake = np.ones((self.batch_size, 1))
        self.labels_disc = np.concatenate([self.labels_fake, self.labels_real], axis=0)


        # path to save model checkpoint during training
        self.checkpoint_path_generator = "./snapshots/cpgen-{epoch:04d}.ckpt"
        self.checkpoint_path_discriminator = "./snapshots/cpdis-{epoch:04d}.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path_generator)

    def compile(self):
        super().compile()

        # Prepare all train metrics
        self.d_train_loss_metric = Mean(name="d_train_loss_metric")
        self.d_train_accuracy_metric = Accuracy(name="d_train_accuracy_metric")
        self.g_train_loss_bce_metric = Mean(name="g_train_bce_loss_metric")
        self.g_train_loss_vgg_metric = Mean(name="g_train_vgg_loss_metric")
        
        # Prepare all validation metrics
        self.d_val_loss_metric = Mean(name="d_val_loss_metric")
        self.d_val_accuracy_metric = Accuracy(name="d_val_accuracy_metric")
        self.g_val_loss_bce_metric = Mean(name="g_val_loss_bce_metric")
        self.g_val_loss_vgg_metric = Mean(name="g_val_loss_vgg_metric")

    def call(self, inputs, **kwargs):
        return None

    @property
    def metrics(self):
        return [self.d_train_loss_metric, self.d_train_accuracy_metric, self.g_train_loss_bce_metric, self.g_train_loss_vgg_metric,
                self.d_val_loss_metric, self.d_val_accuracy_metric, self.g_val_loss_bce_metric, self.g_val_loss_vgg_metric]

    def build_model(self):
        # Create models of both networks
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()

        # Optimizer settings
        self.optimizer_gen = Adam(lr=self.generator_lr, beta_1=self.beta_1, beta_2=self.beta_2)
        self.optimizer_disc = Adam(lr=self.discriminator_lr, beta_1=self.beta_1, beta_2=self.beta_2)

        # Compile discriminator and generator
        self.discriminator.compile(optimizer=self.optimizer_disc, loss=self.loss, metrics=['accuracy'])
        self.generator.compile(optimizer=self.optimizer_gen, loss=[self.loss, self.loss_gen])
        
        # Print out models and save their structures to image file
        self.generator.summary()
        self.discriminator.summary()
        plot_model(self.generator, to_file="gen_model.png", show_shapes=True, show_dtype=True, show_layer_names=True)
        plot_model(self.discriminator, to_file="disc_model.png", show_shapes=True, show_dtype=True, show_layer_names=True)

    def create_generator(self):
        layer_outputs = [] # Outputs of last convolution in the "left side" of U-net, which will be copied to "right side" of U-net

        grayscale_img = Input(shape=(self.resolution, self.resolution, 1)) # Grayscale image
        gen = ConvSN2D(self.filters_gen, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding="same")(grayscale_img)
        gen = ConvSN2D(self.filters_gen, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding="same")(gen)
        if self.attention_res == self.resolution:
            gen = SelfAttention(self.filters_gen)(gen)
        layer_outputs.append(gen) # Will be connected to output layer later

        filters = self.filters_gen * 2
        resolution = self.resolution // 2

        # Create all layers in "left side" of U-net
        while resolution >= GENERATOR_MIN_RESOLUTION:
            gen = AveragePooling2D()(gen)
            gen = ConvSN2D(filters, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding="same")(gen)
            gen = ConvSN2D(filters, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding="same")(gen)
            if self.attention_res == resolution:
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
            gen = ConvSN2D(filters, kernel_size=2, kernel_initializer='he_normal', padding="same")(gen)

            # Concatenate the upscaled previous layer output with "left tree" output with the same resolution
            concat = Concatenate()([left_side_out, gen])

            # Convolution block
            gen = ConvSN2D(filters, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding="same")(concat)
            gen = ConvSN2D(filters, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding="same")(gen)
            if self.attention_res == resolution:
                gen = SelfAttention(filters)(gen)

            # Next layer has twice the resolution and half the filters
            filters //= 2
        
        # Change the number of feature maps to 3, so we get final RGB image
        output_gen = ConvSN2D(3, kernel_size=1, activation="tanh", kernel_initializer='he_normal', padding="same")(gen)

        # Create the generator model
        generator_model = Model(inputs=grayscale_img, outputs=output_gen)
        return generator_model
            
    def create_discriminator(self):
        rgb_img = Input(shape=(self.resolution, self.resolution, 3)) # RGB image

        # Convolutional input block with attention
        disc = ConvSN2D(self.filters_disc, strides=2, kernel_size=3, kernel_initializer='he_normal', padding="same")(rgb_img)
        disc = LeakyReLU(0.2)(disc)
        disc = ConvSN2D(self.filters_disc, kernel_size=3, kernel_initializer='he_normal', padding="same")(disc)
        disc = LeakyReLU(0.2)(disc)
        if self.attention_res == self.resolution:
            disc = SelfAttention(self.filters_disc)(disc)

        resolution = self.resolution // 2
        filters = self.filters_disc * 2
        while resolution > 4:
            disc = ConvSN2D(filters, strides=2, kernel_size=3, kernel_initializer='he_normal', padding="same")(disc)
            disc = LeakyReLU(0.2)(disc)
            disc = ConvSN2D(filters, kernel_size=3, kernel_initializer='he_normal', padding="same")(disc)
            disc = LeakyReLU(0.2)(disc)
            if self.attention_res == resolution:
                disc = SelfAttention(filters)(disc)
            
            resolution //= 2 # Halve the resolution
            filters *= 2 # Twice the filters
        
        # Output block
        disc = ConvSN2D(filters, kernel_size=4, kernel_initializer='he_normal', padding="valid")(disc)
        disc = LeakyReLU(0.2)(disc)
        disc = Flatten()(disc)
        prob = Dense(1, activation='sigmoid')(disc)

        # Build the discriminator model
        discriminator_model = Model(inputs=rgb_img, outputs=prob)
        return discriminator_model

    # ImageNet (http://image-net.org/)
    def load_dataset(self, dataset_name):
        self.dataset = Dataset(self.resolution)
        if dataset_name == "ImageNet":
            self.dataset.load_imagenet()
        else:
            print("ERROR: Selected invalid dataset.")
            sys.exit(1)
    
    # TODO: Colorize images that were placed in 'images_to_colorize' folder and quit, colorized images are placed into 'results' folder
    def colorize_selected_images(self):
        my_images_to_colorize_path = os.path.abspath(os.getcwd()) + "/images_to_colorize/*" # load images from user to colorize

        for file in glob.glob(my_images_to_colorize_path):
          image = Image.open(file)
          image = image.resize((self.resolution, self.resolution)) # resize to resolution on which was net trained 
          data = np.asarray(image)
          grayscale_img = rgb2gray(data) 
          print(grayscale_img.shape)
          plt.imshow(grayscale_img, cmap="gray")
          plt.show()

        
    
    # Initialize variables regarding training continuation 
    def prepare_snapshot_load(self, starting_epoch, weights_path):
        self.starting_epoch = starting_epoch
        self.weights_path = weights_path
        self.load_weights = True

    def train(self):
        # https://www.tensorflow.org/tensorboard/get_started
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit

        # Create Tensorboard callback and set Tensorboard log folder name
        tensorboard_callback = TensorBoard(log_dir=self.logdir, histogram_freq=1, update_freq=self.output_frequency, write_images=True)
        
        # save whole model during training for each epoch 
        cp_callback_generator = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_path_generator, 
                verbose=1, 
                save_weights_only=True,
                save_freq=1*self.batch_size)
        
        cp_callback_generator.set_model(self.generator)

        cp_callback_discriminator = tf.keras.callbacks.ModelCheckpoint(
                filepath=self.checkpoint_path_discriminator, 
                verbose=1, 
                save_weights_only=True,
                save_freq=1*self.batch_size)
        
        cp_callback_discriminator.set_model(self.discriminator)

        # Get train and validation batch generators
        train_gen = self.dataset.batch_provider(self.batch_size)
        val_gen = self.dataset.batch_provider(self.batch_size, train=False)

        # Add train and val image sample to Tensorboard
        train_gt_sample, _, train_bw_sample = next(self.dataset.batch_provider(EXAMPLE_COUNT, convert_range=False))
        val_gt_sample, _, val_bw_sample = next(self.dataset.batch_provider(EXAMPLE_COUNT, train=False, convert_range=False))
        self.file_writer = summary.create_file_writer(self.logdir+"/train/plots")
        with self.file_writer.as_default(step=0):
            summary.image("Training data ground truth examples", plot_to_image(image_grid(train_gt_sample)), max_outputs=EXAMPLE_COUNT)
            summary.image("Training data black & white examples", plot_to_image(image_grid(train_bw_sample, cmap="gray")), max_outputs=EXAMPLE_COUNT)
            summary.image("Validation data ground truth examples", plot_to_image(image_grid(val_gt_sample)), max_outputs=EXAMPLE_COUNT)
            summary.image("Validation data black & white examples", plot_to_image(image_grid(val_bw_sample, cmap="gray")), max_outputs=EXAMPLE_COUNT)
        self.file_writer.close()
        
        # Create ResultsGenerator callback, which periodically saves the network outputs
        results_callback = ResultsGenerator(self.generator, self.dataset, self.logdir, tensorboard_callback, self.output_count, self.output_frequency)

        # Batches per epoch (we have to calculate this manually, because batch provider is running infinitety)
        epoch_batches = self.dataset.train_count//self.batch_size
        
        # Train the model
        self.fit(train_gen, batch_size=self.batch_size, 
                            epochs=self.epochs, 
                            initial_epoch=self.starting_epoch,
                            callbacks=[results_callback, tensorboard_callback, cp_callback_generator, cp_callback_discriminator], 
                            steps_per_epoch=epoch_batches,
                            validation_data=self.dataset.val_data,
                            validation_steps=self.val_batches)

        # save final processed trained model 
        #self.save(os.path.abspath(os.getcwd()) + "/snapshots/my_model")
        # save final weights of trained model 
        self.generator.save_weights(os.path.abspath(os.getcwd()) + "/snapshots/saved_final_weights_generator")
        self.discriminator.save_weights(os.path.abspath(os.getcwd()) + "/snapshots/saved_final_weights_discriminator")

    def generator_step(self, input_image, training=True):
        # Run generator
        generated_images = self.generator(input_image, training=training)
        predictions_gen = self.discriminator(generated_images, training=training)
        g_loss_bce = BinaryCrossentropy()(self.labels_real, predictions_gen)

        # We use VGG to force network to output the same image, but colored,
        # so we take input grayscale images and output image converted to grayscale and measure the loss
        # Images need to have 3 channels in order to work with VGG, so we stack the single grey channel 
        grayscale_images_vgg = tf.image.grayscale_to_rgb(K.cast(input_image, "float32"))
        generated_images_vgg = tf.image.rgb_to_grayscale(generated_images)
        generated_images_vgg = tf.image.grayscale_to_rgb(generated_images_vgg)
        g_loss_vgg = tf.reduce_mean(perceptual_loss(grayscale_images_vgg, generated_images_vgg))
        
        return g_loss_bce, g_loss_vgg 

    # Inspiration: https://keras.io/examples/generative/dcgan_overriding_train_step/
    def train_step(self, data):
        # Get image and labels
        real_images, labels, grayscale_images = data

        # Generate RGB images from grayscale ground truth
        generated_images = self.generator(grayscale_images)

        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Train discriminator
        with GradientTape() as tape:
            predictions_disc = self.discriminator(combined_images)
            d_loss = BinaryCrossentropy(label_smoothing=0.1)(self.labels_disc, predictions_disc)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.optimizer_disc.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Get new batch of images for generator training
        _, labels, grayscale_images = next(self.dataset.batch_provider(self.batch_size))
        
        # Train generator
        with GradientTape() as tape:
            g_loss_bce, g_loss_vgg = self.generator_step(grayscale_images)
            g_loss = g_loss_bce + g_loss_vgg
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.optimizer_gen.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_train_loss_metric.update_state(d_loss)
        self.d_train_accuracy_metric.update_state(np.around(self.labels_disc), math.round(predictions_disc))
        self.g_train_loss_bce_metric.update_state(g_loss_bce)
        self.g_train_loss_vgg_metric.update_state(g_loss_vgg)

        return {
            "d_train_loss": d_loss,
            "d_train_accuracy": self.d_train_accuracy_metric.result(),
            "g_train_loss_vgg": g_loss_vgg,
            "g_train_loss_bce": g_loss_bce
        }
    
    def test_step(self, data):
        # Get input and labels for networks
        # Since we cant use generator for validation data, we have to convert our data manually
        real_images, labels = data
        real_images = real_images.numpy()
        
        # Create grayscale data
        grayscale_images = convert_all_imgs_to_grayscale(real_images)
        
        # Scale data to <-1;1>
        real_images = (real_images - 127.5) / 127.5 
        grayscale_images = (grayscale_images - 127.5) / 127.5

        # Convert ground truth to float32 tensor so it's the same type as generator output and can be concatenated to make discriminator input
        real_images = tf.convert_to_tensor(real_images, "float32")
        generated_images = self.generator(grayscale_images, training=False)
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Run discriminator
        predictions_disc = self.discriminator(combined_images, training=False)
        d_loss = BinaryCrossentropy()(self.labels_disc, predictions_disc)

        # Run generator
        g_loss_bce, g_loss_vgg = self.generator_step(grayscale_images)

        # Update metrics
        self.d_val_loss_metric.update_state(d_loss)
        self.d_val_accuracy_metric.update_state(self.labels_disc, predictions_disc)
        self.g_val_loss_bce_metric.update_state(g_loss_bce)
        self.g_val_loss_vgg_metric.update_state(g_loss_vgg)

        return {
            "d_val_loss": self.d_val_loss_metric.result(),
            "d_val_accuracy": self.d_val_accuracy_metric.result(),
            "g_val_loss_bce": self.g_val_loss_bce_metric.result(),
            "g_val_loss_vgg": self.g_val_loss_vgg_metric.result()
        }