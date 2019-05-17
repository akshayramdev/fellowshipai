# Classification of different models of cars
Explored different datasets and class labels found in https://ai.stanford.edu/~jkrause/cars/car_dataset.html.

## Motivation
Our goal for this project was to create a model to predict the make, model, and year of different cars.  The dataset we used had 196 different classes and it was split into 8,144 training images and 8,041 testing images.  The dataset is provided by Stanford.  It can be viewed at https://ai.stanford.edu/~jkrause/cars/car_dataset.html.  The dataset also came with bounds which we used to crop the images.

## Fast.ai Model
We used the fast.ai library to train the model with several different architectures.  We also used TTA and normalized the images on imagenet_stats.  We found that mixup and custom data augmentations didn't improve the model accuracy and in some cases descreased it.  Our best results were obtained with Densenet-161 trained on 28 epochs.  this model was ~90-91% accuracte.  The notebook for these models is located here: https://github.com/fellowship/platform-demos3/blob/master/car_classification/cars_model.ipynb

## Limitations
This dataset had many classes that were virtually identical.  The most common example of this is cars of the same model but slightly different make or year.  These cars would be hard for the average human to differentiate.  We found that finer grained architectures architecuteres performed better because of this.
