# Real Estate Rooms Classification

The task is to build a classification model which can initially classify the room images into 7 base classes - kitchen, bathroom, bedroom, dining_room, living_room, Exterior and Interior. As next step, the goal is to automatically tag these images with rich attributes for each room image. For e.g Kitchen: appliances, kitchen_island, Interior: Laundry Room, basement, attic, etc, Living Room: fireplace, TV, etc.  

## Dataset

We use the unlabelled dataset collected by previous cohort by crawling redfin website. The data is stored in terms of 19 batch wise folders, where each folder containes 1000 listings and each listing contains ~20 room images. We choose one batch of images ~ 20000 images, move them single folder and shuffle for variation in training

## Approach

1) we first labeled ~4K images through platform.ai and trained them through platform.  
- Most of other classes are easily identifiable except *Interior*. We label *Interior* rooms as the ones which have features such as - attic, washer_dryer room, study tables, clothes, children room, staircases, entrances, foyers,etc. In cases where images have no furnitures, we choose interior as one which is not visually identifiable into other categories, like cross-section images of multiple room types, a small balcony, images with sloped ceiling, etc.
2) We build fast.ai based model using densenet121 architecture. We used transfer learning approach and trained similar ~4K images obtained from platform. We also experimented on different resnet architectures.Additionally, we also implemented data augmentation

## Current Result

- Platform ai model gave us an accuracy of ~87%.
- Our fast.ai based model resulted in ~92% accuracy

## Additional work

- Label more images
- Multi-label classification


