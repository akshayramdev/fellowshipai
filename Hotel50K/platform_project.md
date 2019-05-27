# Hotel scene classification (Wild vs Studio)

We used a subset of the newly released hotel50K dataset to identify whether a hotel image is taken from a travel website (studio) or is taken from TraffickCam (wild). On Platform.AI, out of 1000 images uploaded we labelled and trained on ~500 with 2 classes (studio, wild). The Platform.AI model achieved 82% accuracy, with distinct arms when we visualize the dataset.

We then split the 500 images which were labeled on Platform.AI into train, validation and test sets (7:2:1) and trained our fast.ai model. The Resnet50 models acheived 100% on the test set.

Collab Notebook:
https://colab.research.google.com/drive/1vEKHoVgVRp6xOXOaFHfsBfjlunSaye8x

Platform App:
https://platform.ai/app/imageTraining/projects/ahJzfnBsYXRmb3JtLWFpLXByb2RyUwsSBFVzZXIiHFl5eDk3ZlVkRzVUa0lJcmFGREtkcUlxYUJOQjIMCxIHUHJvamVjdCIgY2ZjODVjNGMyMDMxNDNjNGFiNTRkYTcxZjk3MTk3MGUM