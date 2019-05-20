# Automatic Pixelwise Labelling of Aerial Imagery
Based on [Inria Aerial Image Labeling Contest](https://project.inria.fr/aerialimagelabeling/contest/)

For Benchmark results of Inria Challenge read the [article](https://hal.inria.fr/hal-01468452/document)

## Team members
Pallavi Allada, Jhansi Anumula, Zoheb Abai

## Dataset
The dataset includes urban settlements over the United States and Austria, and is labeled into building and not building classes. Contrary to all previous datasets, the training and test sets are split by city instead of excluding random pixels or tiles. 

![Images with their reference](https://github.com/fellowship/platform-demos3/blob/master/InriaAerialImages/Dataset%20images%20with%20reference.png)

|Train| Tiles | Test | Tiles|
|:------|:----|-----:|-----:|
|Austin, TX | 36 |Bellingham, WA |36|
|Chicago, IL | 36 | San Francisco, CA| 36 |
|Kitsap County, WA | 36 | Bloomington, IN |36|
|Vienna, Austria | 36 |Innsbruck, Austria |36|
|West Tyrol, Austria | 36 | East Tyrol, Austria| 36|
|Total | 180 | Total | 180|

Validation set is created from Training dataset by separating the first five tiles of each area from the training set (e.g. Austin{1-5}).

## Problem Statement
In a given aerial image, identify and mark the presence of buildings i.e. classify at a pixel level of the aerial images if it belongs to the building class or not. The challenge is to design methods that generalize to different areas of the earth, considering the important intra-class variability encountered over large geographic extents. 

## Steps 

### 1. Preprocessing the data: 
Two approaches are implemented to preprocess the image files due to large size of images (5k X 5k).
1. Resize the images offline to a lower dimension (512x512) using bilinear interpolation technique and save the resulting images to drive, inorder to avoid repeated conversions.
2. Generate patches of lower dimension (250x250) with no overlapping pixels with a sliding window logic both for .jpg and .tif formats, and save the generated patches to drive, to avoid repeated operations.

### 2. Model building
Once we have the preprocessed data, we apply few image augmentations, import pretrained resnet-18 model as encoder and train the UNet model. We use two evaluation metrics to assess the performance of our model on the dataset. First, the accuracy,
which is defined as the percentage of correctly classified pixels. Secondly, the intersection over union (IoU) of the positive (building) class, which is defined as the number of pixels labeled as building in both the prediction and the reference, divided by the number of pixels labeled as building in the prediction or the reference IOU and accuracy. For training the model we define a combined loss function which is summation of Dice-coefficient loss function and Crossentropy loss function.

#### Approach 1 : Resized to 512 and Resnet-34 as encoder
Our model got max accuracy of 95% but IOU of 50%, which did not cross the baseline model of INRIA mentioned here. It might be due to huge information loss while resizing the images.

#### Approach 2 : Sliding Window Patches of size 250 and Resnet-18 as encoder
For JPEG images, our model got max accuracy of 95% and IOU of 75%.
For TIFF images, our model got max accuracy of 96% and IOU of 86%. We are submitting the predictions on test images using later model for the challenge as of 20th May 2019.
