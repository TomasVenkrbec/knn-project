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
from dataset import Dataset
from tensorflow import GradientTape, math, summary
from datetime import datetime
from utils import plot_to_image, image_grid, perceptual_loss
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import sys

GENERATOR_MIN_RESOLUTION = 8 # Resolution in "deepest" layer of generator U-net
EXAMPLE_COUNT = 25 # Number of example images in Tensorboard

class DeOldify(Model):
    def __init__(self, 
                resolution=64, 
                filters_gen=32, 
                filters_disc=16, 
                generator_lr=0.0001, 
                discriminator_lr=0.0004,
                batch_size=2,
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
        self.epochs = epochs
        self.output_frequency = output_frequency
        self.output_count = output_count
        self.logdir = logdir + "/" + datetime.now().strftime("%Y%m%d-%H%M%S") 

        # Network-related settings
        self.resolution = resolution # Resolution of train images
        self.filters_gen = filters_gen # Number of convolutional filters in first and last layer of generator
        self.filters_disc = filters_disc # Number of convolutional filters in first layer of discriminator
        self.loss = BinaryCrossentropy()

        # Hyperparameters
        self.generator_lr = generator_lr
        self.discriminator_lr = discriminator_lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def compile(self):
        super().compile()

        # Prepare all metrics
        self.d_loss_metric = Mean(name="d_loss_metric")
        self.d_accuracy_metric = Accuracy(name="d_accuracy_metric")
        self.g_loss_metric = Mean(name="g_loss")

    def call(self, inputs, **kwargs):
        return None

    @property
    def metrics(self):
        return [self.d_loss_metric, self.d_accuracy_metric, self.g_loss_metric]

    def build_model(self):
        # Create models of both networks
        self.generator = self.create_generator()
        self.discriminator = self.create_discriminator()

        # Optimizer settings
        self.optimizer_gen = Adam(lr=self.generator_lr, beta_1=self.beta_1, beta_2=self.beta_2)
        self.optimizer_disc = Adam(lr=self.discriminator_lr, beta_1=self.beta_1, beta_2=self.beta_2)

        # Compile discriminator and generator
        self.discriminator.compile(optimizer=self.optimizer_disc, loss=self.loss, metrics=['accuracy'])
        self.generator.compile(optimizer=self.optimizer_gen, loss=[self.loss, perceptual_loss])
        
        # Print out models and save their structures to image file
        self.generator.summary()
        self.discriminator.summary()
        plot_model(self.generator, to_file="gen_model.png", show_shapes=True, show_dtype=True, show_layer_names=True)
        plot_model(self.discriminator, to_file="disc_model.png", show_shapes=True, show_dtype=True, show_layer_names=True)

    def create_generator(self):
        layer_outputs = [] # Outputs of last convolution in the "left side" of U-net, which will be copied to "right side" of U-net

        grayscale_img = Input(shape=(self.resolution, self.resolution, 1)) # Grayscale image
        gen = ConvSN2D(self.filters_gen, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding="same")(grayscale_img)
        gen = SelfAttention(self.filters_gen)(gen)
        gen = ConvSN2D(self.filters_gen, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding="same")(gen)
        gen = SelfAttention(self.filters_gen)(gen)
        layer_outputs.append(gen) # Will be connected to output layer later

        filters = self.filters_gen * 2
        resolution = self.resolution // 2

        # Create all layers in "left side" of U-net
        while resolution >= GENERATOR_MIN_RESOLUTION:
            gen = AveragePooling2D()(gen)
            gen = ConvSN2D(filters, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding="same")(gen)
            gen = SelfAttention(filters)(gen)
            gen = ConvSN2D(filters, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding="same")(gen)
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
            gen = SelfAttention(filters)(gen)
            
            # Concatenate the upscaled previous layer output with "left tree" output with the same resolution
            concat = Concatenate()([left_side_out, gen])

            # Convolution block
            gen = ConvSN2D(filters, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding="same")(concat)
            gen = SelfAttention(filters)(gen)
            gen = ConvSN2D(filters, kernel_size=3, activation="relu", kernel_initializer='he_normal', padding="same")(gen)
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
        disc = SelfAttention(self.filters_disc)(disc)
        disc = ConvSN2D(self.filters_disc, kernel_size=3, kernel_initializer='he_normal', padding="same")(disc)
        disc = LeakyReLU(0.2)(disc)
        disc = SelfAttention(self.filters_disc)(disc)

        resolution = self.resolution // 2
        filters = self.filters_disc * 2
        while resolution > 4:
            disc = ConvSN2D(filters, strides=2, kernel_size=3, kernel_initializer='he_normal', padding="same")(disc)
            disc = LeakyReLU(0.2)(disc)
            disc = SelfAttention(filters)(disc)
            disc = ConvSN2D(filters, kernel_size=3, kernel_initializer='he_normal', padding="same")(disc)
            disc = LeakyReLU(0.2)(disc)
            disc = SelfAttention(filters)(disc)
            
            resolution //= 2 # Halve the resolution
            filters *= 2 # Twice the filters
        
        # Output block
        disc = ConvSN2D(filters, kernel_size=4, kernel_initializer='he_normal', padding="valid")(disc)
        disc = LeakyReLU(0.2)(disc)
        disc = SelfAttention(filters)(disc)
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
        pass
    
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
        
        # Get train and validation batch generators
        train_gen = self.dataset.batch_provider(self.batch_size)
        val_gen = self.dataset.batch_provider(self.batch_size, train=False)

        # Add train and val image sample to Tensorboard
        train_gt_sample, _, train_bw_sample = next(self.dataset.batch_provider(EXAMPLE_COUNT))
        val_gt_sample, _, val_bw_sample = next(self.dataset.batch_provider(EXAMPLE_COUNT, train=False))
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
        self.fit(train_gen, batch_size=self.batch_size, epochs=self.epochs, callbacks=[results_callback, tensorboard_callback], steps_per_epoch=epoch_batches)

    # Inspiration: https://keras.io/examples/generative/dcgan_overriding_train_step/
    def train_step(self, data):
        # Get image and labels
        real_images, labels, grayscale_images = data

        # Generate RGB images from grayscale ground truth
        generated_images = self.generator(grayscale_images)

        # Combine them with real images
        combined_images = tf.concat([generated_images, K.cast(real_images, dtype="float32")], axis=0)

        # Create labels for discriminator
        labels_real = np.zeros((self.batch_size, 1))    
        labels_fake = np.ones((self.batch_size, 1))
        labels_disc = np.concatenate([labels_fake, labels_real], axis=0)

        # Add random noise to labels
        labels_disc += 0.05 * np.random.uniform(size=labels_disc.shape)

        # Train discriminator
        with GradientTape() as tape:
            predictions_disc = self.discriminator(combined_images)
            d_loss = BinaryCrossentropy()(labels_disc, predictions_disc)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.optimizer_disc.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        # Train generator in combined model
        real_images, labels, grayscale_images = next(self.dataset.batch_provider(self.batch_size))
        with GradientTape() as tape:
            generated_images = self.generator(grayscale_images)
            predictions_gen = self.discriminator(generated_images)
            g_loss_bce = BinaryCrossentropy()(labels_real, predictions_gen)
            g_loss_vgg = perceptual_loss(real_images, generated_images)
            g_loss = math.add(g_loss_bce, g_loss_vgg)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.optimizer_gen.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.d_accuracy_metric.update_state(np.around(labels_disc), math.round(predictions_disc))
        self.g_loss_metric.update_state(g_loss)

        return {
            "d_loss": self.d_loss_metric.result(),
            "d_accuracy": self.d_accuracy_metric.result(),
            "g_loss": self.g_loss_metric.result()
        }