# Face detection using Platform.ai and ArcFace implementation in Pytorch & Fast.ai

### Team Member
Harvey Nguyen, Geoffrey Harrazi, Kien Vu

### Platform.ai model
[Link to model on platform.ai - archived 95% accuracy with three classes: 000289, 000141, 000218](https://platform.ai/app/imageTraining/projects/ahJzfnBsYXRmb3JtLWFpLXByb2RyUwsSBFVzZXIiHE5XSmc0dE1obnZQdnFKUlk1eGoyNmNVUFI1SjIMCxIHUHJvamVjdCIgMTIzYTMwZWFkMzdhNDE2ZWI0N2ZkOGM2YmY0MTNhMDIM
) 

## The Project

[Addictive Angular Loss Margin (ArcFace) paper](https://arxiv.org/abs/1801.07698)

The aim of the project is to build a face classification model based on the VGGFace2 dataset and the CelebA dataset with the ArcFace loss function. According to the paper, ArcFace archives state-of-the-art accuracy with multiple large facial recognition datasets. ArcFace provides improvement over other loss functions (SphereFace, Intraloss, Triplet Loss) by penalising the marginal distance between the sample and the centre of the class on hypersphere.

After training with ArcFace, the test results will be compared with Platforn.ai prediction. Due to large sample size (2.9 GB of testing data with 500 identities from VGGFace2, it is challenging to test all at once. The first step is to test the model with a subset of VGGFace2 with 100 identities for both Platform.ai and ArcFace. Once the models are tested, full dataset will be used for training. Further directions such as using Adacos, which provides state-of-the-art result, will be experimented.

## Dataset

[VGGFace2 Dataset](http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/index.html#about)

VGGFace2 is one of the most comprehenssive facial image dataset with 9131 identities, divided into train set (8631 identities) and test set (500 test set). The identities between train set and test set are disjointed.

Since the dataset is large (over 39 GB), a subset of the dataset is extracted to test ArcFace model and Platform.ai. The sample dataset consists of 100 random identities.

## Aproach

1.  Using Platform.ai with the sample dataset from VGGFace2.
2.  Testing the sample dataset with the pretrained ResNet50 model to establish baseline measurement.
3.  Implementing ArcFace and compare with Platform.ai as well as ResNet50 baseline model.
