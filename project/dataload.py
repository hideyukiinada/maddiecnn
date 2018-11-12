#!/usr/bin/env python
'''Load image files containing images of my 4 dogs.

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT license"
__email__ = "hideyuki@gmail.com"
'''

import os
import logging

from pathlib import Path
from PIL import Image
import numpy as np
import random

from project.config import load_config
from project.maddiecnncommon import DogClassIndex
from project.maddiecnncommon import DogClassMarker

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))  # Change the 2nd arg to INFO to suppress debug logging
config = load_config()

def load_image_data(test_set_only=False):
    """Load image files from the dataset directory

    Parameters
    ----------
    test_set_only: bool
        Load test set from a separate directory

    Returns
    -------
    image_list: list
        List of images
    image_class_list: list
        List of class vectors containing ground truth values
    file_name_list: list
        Image file names ordered to match the sequence of image_list
    """

    if test_set_only:
        dataset_path = Path(config["test_set_directory"])
    else:
        dataset_path = Path(config["dataset_directory"])

    log.debug("Loading data set from %s" % (dataset_path))

    image_list = list()
    image_class_list = list()

    files = list(dataset_path.glob("*.jpg"))
    num_files = len(files)
    log.debug("Number of image files found: %d" % (num_files))

    file_name_list = list()

    dataset_size = 0
    for i, f in enumerate(files):
        image_pil = Image.open(f)
        image_size = image_pil.size

        if image_size[0] != config["image_width"]:
            log.warning("Unexpected width %d for image file %s. Expecting: %d" % (image_size[0], f, config["image_width"]))
            continue

        if image_size[1] != config["image_height"]:
            log.warning("Unexpected height %d for image file %s. Expecting: %d" % (image_size[1], f, config["image_height"]))
            continue

        # Extract classes from the file name
        name = f.name
        underscore_pos = name.find("_")
        if underscore_pos < 0:
            log.warning("Invalid image file name. Missing classification marker for file %s" % (f))
            continue
        classes = name[0:underscore_pos]
        # log.debug("Classes: %s for file %s" % (classes, name))

        class_vector = np.zeros((config["number_of_output_classes"]))
        if DogClassMarker.AIMEE in classes:
            class_vector[DogClassIndex.AIMEE] = 1.0
        if DogClassMarker.MADDIE in classes:
            class_vector[DogClassIndex.MADDIE] = 1.0
        if DogClassMarker.OLIVIA in classes:
            class_vector[DogClassIndex.OLIVIA] = 1.0
        if DogClassMarker.PINK in classes:
            class_vector[DogClassIndex.PINK] = 1.0
        # log.debug("Class vector %s" % (repr(class_vector)))

        data = np.array(image_pil)  # Convert from PIL image to numpy array
        image_list.append(data)
        image_class_list.append(class_vector)

        file_name_list.append(name)

        dataset_size = dataset_size + 1

        if i + 1 >= config["max_number_of_files_to_load"] and config["max_number_of_files_to_load"] != -1:
            break

    log.debug("Number of image files loaded: %d" % (dataset_size))

    return image_list, image_class_list, file_name_list

def load_training_set_and_test_set(test_data_ratio=0.2):

    """Load image files from the dataset directory

    Parameters
    ----------
    test_data_ratio: float
        Ratio of test data size in the total dataset size

    Returns
    -------
    x: numpy array
        Training set data
    y: numpy array
        Training set ground truth values
    x_test: numpy array
        Test set data
    y_test: numpy array
        Test set ground truth values
    file_name_training_list: list
        Training set image file names ordered to match the sequence of training set
    file_name_test_list: list
        Test set image file names ordered to match the sequence of training set
    """

    image_list, image_class_list, file_name_list = load_image_data()

    dataset_size = len(image_list)

    file_name_training_list = list()
    file_name_test_list = list()

    training_dataset_size = int(dataset_size * (1 - test_data_ratio))
    test_dataset_size = dataset_size - training_dataset_size

    x = np.zeros((training_dataset_size, config["image_height"], config["image_width"],
                  config["number_of_channels_in_image"]), dtype='float32')
    y = np.zeros((training_dataset_size, config["number_of_output_classes"]))

    x_test = np.zeros((test_dataset_size, config["image_height"], config["image_width"],
                       config["number_of_channels_in_image"]), dtype='float32')
    y_test = np.zeros((test_dataset_size, config["number_of_output_classes"]))

    # Shuffle data
    random_index_array = list(range(dataset_size))
    random.shuffle(random_index_array)  # shuffle data inplace

    for i in range(training_dataset_size):
        single_image = image_list[random_index_array[i]]
        x[i] = single_image / 255
        y[i] = image_class_list[random_index_array[i]]
        file_name_training_list.append(file_name_list[random_index_array[i]])

    for i in range(test_dataset_size):
        single_image = image_list[random_index_array[i + training_dataset_size]]
        x_test[i] = single_image / 255
        y_test[i] = image_class_list[random_index_array[i + training_dataset_size]]
        file_name_test_list.append(file_name_list[random_index_array[i + training_dataset_size]])

    log.debug("Data loaded")
    return x, y, x_test, y_test, file_name_training_list, file_name_test_list


def load_test_set():

    """Load image files from the test set directory


    Returns
    -------
    x_test: numpy array
        Test set data
    y_test: numpy array
        Test set ground truth values
    file_name_test_list: list
        Test set image file names ordered to match the sequence of test set
    """

    image_list, image_class_list, file_name_list = load_image_data(test_set_only=True)

    dataset_size = len(image_list)

    file_name_test_list = list()

    x_test = np.zeros((dataset_size, config["image_height"], config["image_width"],
                       config["number_of_channels_in_image"]), dtype='float32')
    y_test = np.zeros((dataset_size, config["number_of_output_classes"]))

    for i in range(dataset_size):
        single_image = image_list[i]
        x_test[i] = single_image / 255
        y_test[i] = image_class_list[i]
        file_name_test_list.append(file_name_list[i])

    log.debug("Test set loaded")
    return x_test, y_test, file_name_test_list
