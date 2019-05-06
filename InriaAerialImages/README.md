# Automatic pixelwise labelling of aerial imagery
Based on the contest: https://project.inria.fr/aerialimagelabeling/

## Problem Statement:
In a given aerial image, identify and mark the presence of buildings i.e. classify at a pixel level of the aerial images if it belongs to the building class or not.


## Steps:

1. EDA of the dataset provided, for download. 
2. The image files are large, hence only a subset of available data will be used for training. 
3. Download other datasets from public sources with no building at all, to be able to test the fast.ai trained model to segment all pixel as not building.

