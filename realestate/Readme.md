**Project Description:**

Here, we intend to build a real estate classification model similar to one build by cohort 13. The aim is to classify the room images into 7 base classes - kitchen, bathroom, bedroom, dining_room, living_room, Exterior and Interior. We use the unlabelled dataset collected by previous team from redfin website. The data was stored in terms of 19 batch wise folders, where each folder contained 1000 listings and each listing contained ~20 images.

For the purpose of labelling, we run prediction using pre-trained model on set of 20000 images of any one batch. We visually correct labeling for half of data. Except Interior, other classes are easily identifiable. We classify Interior rooms as the ones which have features such as - attic, washer_dryer room, study tables, children room, staircases, entrances, foyers,etc. In cases where images have no furnitures, we choose interior as one which is not visually identifiable into other categories, like images with windows or rooms with sloped ceiling

Additionally, we use annoy to label data. This is done by finding similar images from unlabelled sample for a given sample of train images. We then use the label of train image in order to label test images.

The train data was laballed from one batch and for validation we use data from another batch. This way, We were able to gather 3000 labelled images. We have trained model on top of pretrained model of previous cohort. With this, we could reach ~80% accuracy.

**Additional Work:**

Increase Label Data size
Perform Data Augmentation

