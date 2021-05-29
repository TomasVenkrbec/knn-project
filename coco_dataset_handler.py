#!/usr/bin/env python
# coding: utf-8

import os
import sys
import shutil
import datetime
import argparse

sys.path.insert(1, './cocoapi-master/PythonAPI')

from pycocotools.coco import COCO

LEN_OF_COCO_IMAGE_NAME = 12


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def parse_args():
    argParser = argparse.ArgumentParser(description='Prepares required images from the COCO dataset.')
    argParser.add_argument('--dinfo', dest='dataset_info', action='store_true', default=False, help='Prints info about categories and supercategories that are available in the provided COCO dataset. Using this argument, no dataset processing will be performed.')
    argParser.add_argument('--dataset', dest='dataset', action='store', help='COCO dataset folder with subfolders "images" and "annotations".')
    argParser.add_argument('--dtype', dest='dataset_type', action='store', default='train2017', help='Type of COCO dataset that defines source files - ie. "train2017"')
    argParser.add_argument('--categories', dest='categories', action='store', default='[cat]', help='Choose categories (or supercategories, or both) of images that should be selected from the dataset as a list, ie. [cat, person, car]')
    argParser.add_argument('--otype', dest='output_type', action='store', default='print', help='Output form of the COCO dataset images: "print" - prints names of images to stdout, "subfolder" - creates a subfolder containing the selected images in the dataset folder.')
    return argParser.parse_args()


def print_info_about_cats(COCO):
    supcats, cats = get_categories(COCO)
    print("PRINTING INFO ABOUT CATEGORIES:")
    print("--------------------------------------------")
    print('COCO supercategories: \n{}\n'.format(supcats))
    print('COCO categories: \n{}'.format(cats))
    print("--------------------------------------------")


def get_categories(COCO):
    cats = COCO.loadCats(COCO.getCatIds())

    supercategories = list(set([cat['supercategory'] for cat in cats]))
    categories = [cat['name'] for cat in cats]

    return supercategories, categories


def get_filenames_by_cats(supcat_names=[], cat_names=[]):
    if supcat_names:
        supercatIds = coco.getCatIds(supNms=supcat_names)
    else:
        supercatIds = []
    if cat_names:
        catIds = coco.getCatIds(catNms=cat_names)
    else:
        catIds = []


    imgIds = []
    for catId in list(set(supercatIds + catIds)):
        imgIds.extend(coco.getImgIds(catIds=catId))
    imgIds = list(set(imgIds))

    return turn_ids_to_names(imgIds)


def turn_id_to_name(imageID):
    imageLen = len(str(imageID))
    imageName = (LEN_OF_COCO_IMAGE_NAME - imageLen) * '0' + str(imageID) + '.jpg'
    return imageName


def create_subfolder(dataset_dir, selected_cats):
    name = 'COCO'
    for cat in selected_cats:
        name += '_' + str(cat)
    name += '_' + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    mydir = os.path.join(dataset_dir, name)
    os.makedirs(mydir)
    return mydir


def turn_ids_to_names(imagesIDs):
    return [turn_id_to_name(x) for x in imagesIDs]


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def string_to_list(string):
    return string.replace(" ", "").strip('][').split(',')


def copy_files_to_subfolder(dataset_folder, target_folder, images_to_copy):
    count = 0
    for file in os.listdir(dataset_folder + '/images'):
        if file in images_to_copy:
            name = os.path.join(dataset_folder + '/images', file)
            if os.path.isfile(name):
                shutil.copy(name, target_folder)
                count += 1
            else:
                print(file + ' does not exist')
    return count


if __name__ == "__main__":
    args = parse_args()
    dataDir = args.dataset
    dataType = args.dataset_type
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

    block_print()
    coco = COCO(annFile)
    enable_print()

    if args.dataset_info:
        print_info_about_cats(coco)
    else:
        super_cats, cats = get_categories(coco)
        selected_cats = string_to_list(args.categories)
        imgNames = get_filenames_by_cats(supcat_names=(intersection(super_cats, selected_cats)), cat_names=(intersection(cats, selected_cats)))

        if args.output_type == 'print':
            print(imgNames)
        elif args.output_type == 'subfolder':
            print("Let's put it into the subfolder...")
            folder = create_subfolder(dataDir, selected_cats)
            print(folder)
            count = copy_files_to_subfolder(dataDir, folder, imgNames)
            print("{} images copied to the target location: {}".format(count, folder))
        else:
            print("I don't know such output type :(")

