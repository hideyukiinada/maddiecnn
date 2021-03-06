# Maddie CNN

# Goals
Identify my 4 dogs in 3000+ photos using Keras-based convolutional neural network.

# Network architecture
Network is defined using Keras' Sequential model.  It uses multiple conv layers as well as dense layers toward the end.  Please see cnn.py for details.

If you run tensorboard, you can also view the architecture during the training time.

# Dataset
Dataset consists of 3679 images.  3303 images contain one or more of the 4 dogs below.  The remaining 376 images do not contain any image of dogs. 

|  Name | Sample photo from the dataset  |
|---|---|
| Aimée | ![Image of Aimee](assets/images/a_02059.jpg_256x256.jpg) |
| Maddie | ![Image of Maddie](assets/images/m_01409.jpg_256x256.jpg) |
| Olivia | ![Image of Olivia](assets/images/o_01106.jpg_256x256.jpg) |
| Pink | ![Image of Pink](assets/images/p_01216.jpg_256x256.jpg) |

## Downloading the dataset
Before you download or use the dataset, please carefully read the readme document in the below repository to make sure that your intended use of data complies with the terms specified.  **The dataset is NOT released under the MIT license.**

The dataset is available for those who will be using the data in a way that is compliant with the terms:
https://github.com/hideyukiinada/maddiecnn_dataset

## Size and format of each image
256 pixel by 256 pixel JPEG.

## Classification criteria
For each dog, if I can say with confidence that a dog is in the photo even if a part of the body or face is missing, then the dog is considered present in the photo and marked as such.

## Image file name convention 
Files are named using the following convention:

<Name prefixes>_<a string that contains 5 digit file ID>.jpg

If one or more dogs are in the photo, each file is marked with one or more of the following prefixes: 

| Prefix | Description
|---|---|
| a | Aimée |
| m | Maddie |
| o | Olivia |
| p | Pink |

  
If a photo does not contain any of the dogs above, the file name will be _<a string that contains 5 digit file ID>.jpg
.

Upon data load, classes get converted to a following vector to store ground truth:

| Vector element | Value |
|---|---|
|0|0 or 1.0|
|1|0 or 1.0|
|2|0 or 1.0|
|3|0 or 1.0|

## Breakdown of dataset
| Type | Count | Note | Directory | 
|---|---|---|---|
| Training set | 2974 | Used to train the model | dataset/256 |
| Validation set | 92 | Used during training to validate.  Files were randomly picked by the training script |  dataset/256 |
| Test set | 613 | Used during testing to test against data that was not in training set |  dataset/256.test |
| Total | 3679 |   |

# Accuracy of prediction
Accuracy was measured by running maddiecnn_predict script against 613 images that are not used in training.
Classes for 516 files completely matched (84.1%).  Partial match was considered incorrect.
If you have any lower/higher accurary result in your environment, I'd love to hear from you.

# Running the scripts
## Requirements
- Python virtualenv with Python 3.5.2 and above (it may work with Python 3.4 and above but I haven't verified).
- Keras
- TensorFlow
- Python Pillow

## Training and Predicting
To train the model, run maddiecnn.  To predict, run maddiecnn_predict.

## Directory structure
In addition to the project directory where script and helper code are located, the following directories are required:

| Directory | Note |
|---|---|
| dataset/256 | Images for training and validation set. I put 3066 files in this directory. |
| dataset/256.test | Images for test set.  You need to place some files that are not in the training set in this directory so that maddiecnn_predict can predict against data that is not in the training set. I put 613 files from the dataset in this directory. |
| log | Store log for TensorBoard to visualize |
| weight | Store weight.  Weight file is automatically saved in hdf5 format at the end of the training. To run prediction, you need this weight file.|

These directories should be at the same level as the 'maddiecnn' directory which is created when you clone the repo.

## Configuring parameters
All configurable parameters are specified in config.json.
