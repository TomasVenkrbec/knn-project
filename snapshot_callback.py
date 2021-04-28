from tensorflow.keras.callbacks import Callback
import os

class SnapshotCallback(Callback):
    def __init__(self, generator, discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.checkpoint_path_generator = "./snapshots/generator_weights.ckpt"
        self.checkpoint_path_discriminator = "./snapshots/discriminator_weights.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path_generator)
        


    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))
        self.generator.save_weights(self.checkpoint_path_generator)
        self.discriminator.save_weights(self.checkpoint_path_discriminator)


    