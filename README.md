# KNN project - Image colorisation
## Intro
Project is based on original [DeOldify](https://github.com/dana-kelley/DeOldify) architecture.

**Authors:**
 - Tomáš Venkrbec ([xvenkr01](mailto:xvenkr01@stud.fit.vutbr.cz))
 - Fořtová Kateřina ([xforto00](mailto:xforto00@stud.fit.vutbr.cz))
 - Dvořák Jiří ([xdvora2u](mailto:xdvora2u@stud.fit.vutbr.cz))

## Results
Examples of colorized images are shown in `results` folder.

## Quickstart guide
 1. Pull this repository to local machine: `git pull https://github.com/TomasVenkrbec/knn-project`
 2. Run the install script: `chmod +x install.sh && . install.sh`
    * For quick access to python virtual environment, use `. path.sh` next time.
 3. Start the main script: `python3 run.py --dataset COCO`
    * For list of available arguments, run the script with `-h` argument
    * For conversion of images in `images_to_colorize` folder, use `--images` and `--load_weights` arguments

## Requirements / Tested on
 - Python 3.8.5
 - Tensorflow 2.4.1
 - [ImageNet](http://image-net.org/) dataset (preferably downscaled 64x64 version)
 - [COCO](https://cocodataset.org/#download) dataset

![Swish swoosh](swish.jpg "Swish swoosh")
