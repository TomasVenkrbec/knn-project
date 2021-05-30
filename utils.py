import matplotlib.pyplot as plt
import io
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow import image, expand_dims
import tensorflow as tf
import numpy as np

class PerceptualLoss:
    def __init__(self, resolution):
        # Source: https://stackoverflow.com/questions/65484420/define-custom-loss-perceptual-loss-in-cnn-autoencoder-with-pre-train-vgg19-tenblock5_conv4']
        selected_layers = ["block2_conv2"]

        vgg = VGG19(weights='imagenet', include_top=False, input_shape=(resolution,resolution,3))
        vgg.trainable = False
        outputs = [vgg.get_layer(l).output for l in selected_layers]
        self.model = Model(vgg.input, outputs)

    @tf.function
    def perceptual_loss(self, input_image, reconstruct_image):
        h1_list = self.model(input_image)
        h2_list = self.model(reconstruct_image)

        rc_loss = 0.0
        for h1, h2 in zip(h1_list, h2_list):
            h1 = K.batch_flatten(h1)
            h2 = K.batch_flatten(h2)
            rc_loss = rc_loss + K.sum(K.square(h1 - h2), axis=-1)

        return rc_loss

# Source: https://www.tensorflow.org/tensorboard/image_summaries#logging_arbitrary_image_data
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    tf_image = image.decode_png(buf.getvalue(), channels=3)
    # Add the batch dimension
    tf_image = expand_dims(tf_image, 0)
    return tf_image

# Source: https://www.tensorflow.org/tensorboard/image_summaries#logging_arbitrary_image_data
def image_grid(images, grid_size=5, cmap=None):
    """Return a grid of images as a matplotlib figure."""
    assert len(images) >= grid_size**2

    # Create a figure to contain the plot.
    figure = plt.figure(figsize=(10,10))
    for i in range(int(grid_size**2)):
        # Start next subplot.
        plt.subplot(int(grid_size), int(grid_size), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.01, hspace=0.01)
        if cmap:
            plt.imshow(images[i], cmap=cmap)
        else:
            plt.imshow(images[i])

    return figure