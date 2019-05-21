# Real Estate Rooms Classification

The task is to build a classification model which can initially classify the room images into 7 base classes - kitchen, bathroom, bedroom, dining_room, living_room, Exterior and Interior. As next step, the goal is to atumatically tag these images with further rich attributes within each class. For e.g Kitchen: appliances, kitchen_island, Interior: Laundry Room, basement, attic, etc, Living Room: fireplace, TV, etc.  

## Dataset

We use the unlabelled dataset collected by previous cohort by crawling redfin website. The data is stored in terms of 19 batch wise folders, where each folder containes 1000 listings and each listing contains ~20 room images.

## Approach

- we first run prediction using *realestate-classification-rn50* pre-trained model on set of 20000 images of any one batch. We then manully labeling for intial 1000-2000 images. 
- Most of other classes are easily identifiable except *Interior*. We label *Interio*r rooms as the ones which have features such as - attic, washer_dryer room, study tables, clothers, children room, staircases, entrances, foyers,etc. In cases where images have no furnitures, we choose interior as one which is not visually identifiable into other categories, like cross-sectio images of multiple room types, a small balcony, images with sloped ceiling, etc.
- Next, we use annoy to label other test images. This is done by finding similar images from unlabelled sample for a given sample of train images. We then use the label of train image in order to label test images.
- Once, we have labelled data, we then train them on either *realestate-classification-rn50* pre-trained model or any pretrained *resnet* model. We do this the fast.ai way, by first applying transfer learning and once we see enough improvement we unfreeze model and train all the layers

## Current Result

Presently, we were able to gather ~4500 labelled images. The train data is from one batch and for validation we use images from another batch. We have trained fasti.ai model on top of *realestate-classification-rn50* pre-trained model. With this, we get **~89% accuracy**.
Additionally, training fast.ai model using densenet121 gave an improvement in the accuracy upto **90.1%**

## Additional work

- Build model on platform.ai
- Label more images
- Retrain again with data augmentation


