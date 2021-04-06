import tensorflow as tf
import argparse
import gc
import subprocess
import os
import sys
from datetime import datetime
from model import DeOldify

def parse_args():
    argParser = argparse.ArgumentParser(description='Train a generative adversarial network model from scratch, continue training from a snapshot or run model on given black and white image.')
    argParser.add_argument('--images', dest='images', action='store_true', default=False, help='Run model on black and white images from \'images_to_colorize\' folder.')
    argParser.add_argument('--load_weights', dest='load_weights', action='store_true', help='Load weights from snapshot')
    argParser.add_argument('--snapshot_path', dest='snapshot_path', action='store', help='Snapshot position, defaults to \'snapshots\' folder.')
    argParser.add_argument('--starting_epoch', dest='starting_epoch', action='store', type=int, default=0, help='Starting epoch when continuing training from saved snapshot.')
    argParser.add_argument('--resolution', dest='resolution', action='store', type=int, help='Resolution of samples from dataset.')
    argParser.add_argument('--filters_gen', dest='filters_gen', action='store', type=int, help='Number of filters in first and last layer of generator.') 
    argParser.add_argument('--filters_disc', dest='filters_disc', action='store', type=int, help='Number of filters in first layer of discriminator.')
    argParser.add_argument('--batch_size', dest='batch_size', action='store', type=int, help='Number of samples in every training step.')  
    argParser.add_argument('--output_frequency', dest='output_frequency', action='store', type=int, help='Number of training batches between training metric exports.')  
    argParser.add_argument('--epochs', dest='epochs', action='store', type=int, help='Number of epochs to train for.')
    argParser.add_argument('--dataset', dest='dataset', action='store', default="ImageNet", choices=['ImageNet'], help='Select dataset to be used. All dataset files need to be placed directly in \'dataset/##NAME##/\' folder, not in any subfolders.')
    argParser.add_argument('--disc_lr', dest='disc_lr', action='store', type=float, help='Learning rate of discriminator network.')
    argParser.add_argument('--gen_lr', dest='gen1_lr', action='store', type=float, help='Learning rate of generator network.')
    argParser.add_argument('--beta_1', dest='beta_1', action='store', type=float, help='First order momentum value, hyperparameter of Adam optimizer.') 
    argParser.add_argument('--beta_2', dest='beta_2', action='store', type=float, help='Second order momentum value, hyperparameter of Adam optimizer.') 
    return argParser.parse_args()

def setup_environment():
    # Force enable garbage collector
    gc.enable()

    # Clear Keras cache, just in case
    tf.keras.backend.clear_session()

def print_info():
    print("Using computer: " + os.uname()[1])
    print("Current time: " + datetime.now().strftime('%d-%m-%Y %H:%M:%S'))
    print("Tensorflow version: " + tf.__version__)

def setup_gpu():
    # Find out whether CUDA-capable GPU is available and if it is, allow Tensorflow to use is
    freeGpu = subprocess.check_output('nvidia-smi -q | grep "Minor\|Processes" | grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"', shell=True)
    if len(freeGpu) == 0:
        print('No free GPU available, running in CPU-only mode!')
    else:
        print("Found GPU: " + str(freeGpu))
    os.environ['CUDA_VISIBLE_DEVICES'] = freeGpu.decode().strip()

def load_weights(model, args):
    if args.snapshot_path is not None: # Snapshot path was selected by user
        weights_path = args.snapshot_path
    else:
        weights_path = "snapshots/"

    if not (os.path.exists(weights_path) and os.path.isfile(weights_path + "discriminator_weights.h5") and os.path.isfile(weights_path + "generator_weights.h5")):
        print(f"ERROR: Can't find files with saved weights in {'selected' if not args.snapshot_path else 'default snapshot'} folder.")
        sys.exit(1)

    # Save weights path and starting epoch in the model
    model.prepare_snapshot_load(args.starting_epoch, weights_path)

if __name__ == "__main__":
    print_info()
    setup_environment()
    setup_gpu()

    # Arguments that are used during model initialization are extracted
    init_args = {}
    args = parse_args()
    for arg in vars(args):
        if getattr(args, arg) is not None and arg not in ["load_weights", "snapshot_path", "starting_epoch", "images", "dataset"]:
            init_args[arg] = getattr(args, arg)

    model = DeOldify(**init_args)

    # Load model if needed
    if args.load_weights == True:
        load_weights(model, args)
    
    # Build model
    model.build_model()

    # Only colorize images selected in arguments
    if args.images:
        model.colorize_selected_images()
    else: # Train the model
        model.load_dataset(args.dataset)
        model.compile()
        model.train()
