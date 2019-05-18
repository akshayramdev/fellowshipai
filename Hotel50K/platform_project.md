# Hotel scene classification

We used a subset of the newly released hotel50K dataset to identify hotel scenes. On Platform.AI, out of 1000 images uploaded we labelled and trained on 500 with 4 classes (one_bed, two_beds, bathroom_with_toilets, bathroom_no_toilets). The Platform.AI model achieved 84% accuracy, with four distinct arms when we visualize the dataset.

We then split the 1000 images which Platform.AI predicted into train, validation and test sets (7:2:1) and trained our fast.ai model. We compared densenet121 and resnet101 models on this dataset and they both achieved 80% accuracy on the test set. Gradcam shows that the model is paying attention to important features such as beds, toilets and sinks in the images. One of the potential confounds for our fast.ai model is the training label quality, as we took the predicted labels from our Platform.AI model to be able to train on the full 1000 images, whose labels may or may not be correct.

Platform App:
https://platform.ai/app/imageTraining/projects/ahJzfnBsYXRmb3JtLWFpLXByb2RyUwsSBFVzZXIiHFl5eDk3ZlVkRzVUa0lJcmFGREtkcUlxYUJOQjIMCxIHUHJvamVjdCIgOGNhY2ExMTA2YjFmNGU2ZTgyZGE4NWNhZDA5ZTMyM2UM