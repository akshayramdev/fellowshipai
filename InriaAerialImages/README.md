# Automatic pixelwise labelling of aerial imagery
Based on the contest: https://project.inria.fr/aerialimagelabeling/

## Problem Statement:
In a given aerial image, identify and mark the presence of buildings i.e. classify at a pixel level of the aerial images if it belongs to the building class or not.

## Team members:
Pallavi Allada, Jhansi Anumula, Zoheb Abai

## Steps:
### 1)Preprocessing the data: 
2 approaches will be taken to preprocess the image files as the images are huge - 5k X 5k.
a) Resize the images offline to a lower dimension and save the resulting images to drive, inorder to avoid repeated conversions.
b) Generate patches of lower dimension with overlapping pixels with a sliding window logic, and save the generated patches to drive, to avoid repeated operations.

### 2)Model building
Once we have the preprocessed data, we should be able to train the UNet model. Metrics to watch out for are IOU and accuracy.
Loss to optimise will be Cross Entropy Loss.



