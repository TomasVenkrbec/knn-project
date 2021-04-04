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
    argParser.add_argument('--images', dest='images', nargs='+', help='Run model on selected black and white image')
    argParser.add_argument('--load_weights', dest='load_weights', action='store_true', help='Load weights from snapshot')
    argParser.add_argument('--snapshot_path', dest='snapshot_path', action='store', help='Snapshot position, defaults to \'snapshots\' folder.')
    argParser.add_argument('--starting_epoch', dest='starting_epoch', action='store', type=int, default=0, help='Starting epoch when continuing training from saved snapshot.')
    argParser.add_argument('--dataset', dest='dataset', action='store', default="ImageNet", choices=['ImageNet', 'CIFAR-100'], help='Select dataset to be used. All dataset files need to be placed directly in \'dataset/##NAME##/\' folder, not in any subfolders.')
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

    # Only colorize images selected in arguments
    if args.images is not None:
        model.colorize_selected_images(args.images)
    else: # Train the model
        model.load_dataset(args.dataset)
        model.train()