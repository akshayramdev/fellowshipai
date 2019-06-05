# Classification: Category of CLothing


## Motivation
Clothing in various cultures reflects many things such as age, social status, lifestyle and gender. Apparel is also an important descriptor in identifying different persons. However, understanding fashion is still a very complicated task for computer vision. The field has applications in social media and e-commerce. We focus on several tasks within the fashion classification umbrella:

1.  Multilabel classification of clothing attributes
2.  Object detection

Depending on the specific application of fashion classification, the most relevant problems to solve will be different. We will focus on optimizing fashion classification for the purposes of annotating images and discovering the temporal fashion patterns.

Our pipeline is as follows: given a query image from streetsytle database,

1.  Detect the person and crop images, record the bounding boxes coordinates,
2.  Predict the clothing attributes.

The dataset we are using is streetsytle27k (http://streetstyle.cs.cornell.edu/). 
More detailed exploration and preprocess work can be find in the preprocess_data.ipynb

## fastai Model
We used the fastai library to train the model with different kind of images such as the whole image as well as the full body crop of the person detected by the YOLO Algorithm) as well as different architectures. We obtained our best results with ResNet34, and the model's accuracy is 90%.


## platform.ai Model
We also built a classification demo on platform.ai that sorts the images into certain category of cloth, which is Suits, beachwear, shirts, tshirts and ladiestop. Initially we tried to add almost all the labels, but it didn't work due to inability to handle multi-label classification of platform.ai and the labels were multi-label for each image. Therefore, we created a demo for simple item types/category.
Here's the [link](https://platform.ai/app/imageTraining/projects/ahJzfnBsYXRmb3JtLWFpLXByb2RyUwsSBFVzZXIiHEdBdUp4dXlTQWdoMkFISGszTlhDN1VzRHJndzEMCxIHUHJvamVjdCIgZDhlOTc5ZTM1MjBiNDk1NjlhOWJiNDhhYjliOWZkNzcM) for platform demo.

## Limitations
The streetStyle dataset has limited number of labels, which is less as compared to similar.ai taxonomy. Therefore, annotating images with the Similar.ai taxonomy becomes a difficult task.
