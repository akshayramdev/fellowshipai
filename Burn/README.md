# Burn classification

## Motivation

Identification of burn severity is an important step in diagnosis and treatment, but standard clinical methods are
only 60-80% accurate[[1]](https://www.sciencedirect.com/science/article/abs/pii/S0305417900001182?via%3Dihub).  Our goal was to create an classifier to predict burn severity based on an image.
We used a dataset curated by cohort 13 of 755 web-scraped images of first, second, and third degree burns and non-burn images of skin, with a randomly selected validation set.

## fastai model

We used the fastai library to train models with several different architectures.  In addition to the default set of data augmentations, we also used vertical reflections and mixup[[2]](https://arxiv.org/abs/1710.09412).  We obtained our best results with DenseNet-161, with test-time augmentation scale=1.05. This model was 94.59% accurate (see burn1.ipynb for details).

## Platform.ai model

We also used the platform.ai tool to train an additional [classifier](https://platform.ai/app/imageTraining/projects/ahJzfnBsYXRmb3JtLWFpLXByb2RyUwsSBFVzZXIiHEZUSXdNOThIazVlT2Q0ajJnWFpQRGV5WG42dTEMCxIHUHJvamVjdCIgMzIxZDVhNzA3OGExNDNiOWFkYmY1NmJiYTI2NDc5NWIM), with accuracy ~83%.

## Limitations

Image classification models require correctly identified images for training.  While both of our classifiers give promising results, the images they were trained on were labeled either by us (in the case of the platform classifier) or by the cohort 13 fellows (in the case of the fastai classifier).  Since medical professionals who are not burn experts may incorrectly classify burns 20-40% of the time, it is reasonable to expect that our labeling errors are at least that high.  This suggests that a major limitation of both models' accuracy is the quality of the training labels, and that to improve either classifier we should obtain more training data with more accurate labels.
