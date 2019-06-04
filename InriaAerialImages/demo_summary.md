# Automatic pixel-wise Aerial Image Labeling

**Aerial** photography is used in cartography, land-use planning, archaeology, environmental studies, power line inspection, surveillance, commercial advertising, conveyancing, and artistic projects.

One of the case study we are working is, to figure out the building from the aerial images. 

# Aerial Image Classification:

For Platform.ai, we uploaded the sliced (250X250 size) train images to label whether building present or absent. 
We labeled few images on platform.ai and trained the model. We achieved an accuracy of 95%. 

[Link to platform.ai model](https://platform.ai/app/imageTraining/projects/ahJzfnBsYXRmb3JtLWFpLXByb2RyUwsSBFVzZXIiHHE3blJTUGlGQmlTbmZvb200SU05QW9jekNlZzEMCxIHUHJvamVjdCIgYTg2NjNjMzNiNTRiNGZhMzhlNTJjYzNkN2EyMGQ0MDMM)

In model trained locally using fastai gave us accuracy of 95%. Here we used CNN with pretrained resnet34 architecture.