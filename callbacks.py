from tensorflow.keras.callbacks import Callback
from tensorflow import summary
from utils import plot_to_image, image_grid
import matplotlib.pyplot as plt
import numpy as np

class ResultsGenerator(Callback):
    def __init__(self, generator, dataset, logdir, tb_callback, img_count=36, output_frequency=50):
        self.generator = generator
        self.dataset= dataset
        self.logdir = logdir
        self.img_count = img_count
        self.output_frequency = output_frequency
        self.tb_callback = tb_callback

        # Generate inputs for generator with respective ground truths, will not change over time
        self.gt_sample, _, self.bw_sample = next(dataset.batch_provider(img_count, train=False))

        # Generate inputs for generator without ground truths, will not change over time
        _, _, self.bw_sample_no_gt = next(dataset.batch_provider(img_count, train=False))

    def on_batch_end(self, batch, logs=None):
        if batch % self.output_frequency == 0:
            # Generate the images 
            generated_images_gt = self.generator.predict(self.bw_sample)
            generated_images_no_gt = self.generator.predict(self.bw_sample)

            # Convert to <0;255> range
            generated_images_gt *= 255 
            generated_images_no_gt *= 255

            # Merge arrays together so that new array alternates between real and fake images
            images_gt = np.empty((self.img_count*2, self.gt_sample.shape[1], self.gt_sample.shape[2], self.gt_sample.shape[3]), dtype=np.uint8)
            images_gt[0::2] = self.gt_sample
            images_gt[1::2] = generated_images_gt

            # Convert to grid and to image
            grid_gt = image_grid(images_gt.astype(np.uint8), np.sqrt(self.img_count))
            grid_no_gt = image_grid(generated_images_no_gt.astype(np.uint8), np.sqrt(self.img_count))

            grid_gt.savefig(f"{self.logdir}/results_gt.png")
            grid_no_gt.savefig(f"{self.logdir}/results_no_gt.png")

            tf_image_gt = plot_to_image(grid_gt)
            tf_image_no_gt = plot_to_image(grid_no_gt)

            with self.tb_callback._train_writer.as_default(step=batch):
                summary.image("RGB images from generator with ground truth", tf_image_gt, max_outputs=self.img_count)
                summary.image("RGB images from generator", tf_image_no_gt, max_outputs=self.img_count)