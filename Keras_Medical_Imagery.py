#Not finished yet, still a work in progresxs
# import the necessary packages
from pyimagesearch import config
from imutils import paths
import random
import shutil
import keras
import os

# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "B:\Malaria-Dataset\cell_images"
 
# initialize the base path to the *new* directory that will contain
# our images after computing the training and testing split
BASE_PATH = "B:\Malaria-Dataset\cell_images\malaria"
 
# derive the training, validation, and testing directories
TRAIN_PATH = os.path.sep.join([BASE_PATH, "training"])
VAL_PATH = os.path.sep.join([BASE_PATH, "validation"])
TEST_PATH = os.path.sep.join([BASE_PATH, "testing"])
 
# define the amount of data that will be used training
TRAIN_SPLIT = 0.8
 
# the amount of validation data will be a percentage of the
# *training* data
VAL_SPLIT = 0.1

# grab the paths to all input images in the original input directory
# and shuffle them

imagePaths = list(paths.list_images(config.ORIG_INPUT_DATASET))
random.seed(42)
random.shuffle(imagePaths)
