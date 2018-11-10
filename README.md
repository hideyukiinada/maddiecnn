# maddiecnn
## Goals
Identify my 4 dogs using convolutional neural network (CNN) from 3000+ photos using Keras-based convolutional neural network.

## Network design
Network is defined using Keras' Sequential model.  It uses multiple conv layers as well as dense layers toward the end.  Please see cnn.py for details.

## Dataset
Dataset consists of 3319 images.  3253 images contain one or more of the 4 dogs below.  The remaining 66 images do not contain any image of dogs. 

|  Name | Sample photo from the dataset  |
|---|---|
| Aimee | ![Image of Aimee](assets/images/a_02059.jpg_256x256.jpg) |
| Maddie | ![Image of Maddie](assets/images/m_01409.jpg_256x256.jpg) |
| Olivia | ![Image of Olivia](assets/images/o_01106.jpg_256x256.jpg) |
| Pink | ![Image of Pink](assets/images/p_01216.jpg_256x256.jpg) |

## Size and format of each image
256 pixel by 256 pixel JPEG.

## Classification criteria
For each dog, if I can say with confidence that a dog is in the photo even if a part of the body or face is missing, then the dog is considered present in the photo and marked as such.

## Image file name convention 
Files are named using the following convention:

<Name prefixes>_<5 digit zero-padded number>.jpg

If one or more dogs are in the photo, each file is marked with one or more of the following prefixes: 

| Prefix | Description
|---|---|
| a | Aimée |
| m | Maddie |
| o | Olivia |
| p | Pink |

  
If a photo does not contain any of the dogs above, the file name will be _<5 digit zero-padded number>.jpg.

Upon data load, classes get converted to a following vector to store ground truth:

| Vector element | Value |
|---|---|
|0|0 or 1.0|
|1|0 or 1.0|
|2|0 or 1.0|
|3|0 or 1.0|
