#!/usr/bin/env python
'''Set up CNN using Keras framework.

Credit
------
    I am using a portion of the code from the CNN implementation in one of the Keras example files.
    Please see the license at the end of this docstring for their license terms.

__author__ = "Hide Inada"
__copyright__ = "Copyright 2018, Hide Inada"
__license__ = "The MIT license"
__email__ = "hideyuki@gmail.com"

=========================================================================================
License and copyright statements for the portion of the source code taken from the Keras
mnist_cnn.py examples code
=========================================================================================
COPYRIGHT

All contributions by François Chollet:
Copyright (c) 2015 - 2018, François Chollet.
All rights reserved.

All contributions by Google:
Copyright (c) 2015 - 2018, Google, Inc.
All rights reserved.

All contributions by Microsoft:
Copyright (c) 2017 - 2018, Microsoft, Inc.
All rights reserved.

All other contributions:
Copyright (c) 2015 - 2018, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.

LICENSE

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
=========================================================================================
End of Keras license
'''

import os
import logging

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from project.config import load_config

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))  # Change the 2nd arg to INFO to suppress debug logging
config = load_config()


def cnn():
    """Convolutional Neural Network implementation

    Returns
    -------
    model: Reference to Keras Sequaltial model
        Set up and compile CNN to classify dogs.  Each photo can have multiple dogs, so this is a multilabel classification.
    """

    input_shape = (config["image_height"], config["image_width"], config["number_of_channels_in_image"])

    model = Sequential()
    model.add(Conv2D(8, kernel_size=(7, 7), padding='same', activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # To 128x128

    model.add(Conv2D(16, kernel_size=(5, 5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # To 64x64

    model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # To 32x32

    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # To 16x16

    model.add(Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # To 8x8

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(config["number_of_output_classes"],
                    activation='sigmoid'))  # Do not use softmax as there are multiple valid labels

    model.compile(loss=keras.losses.binary_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    return model
