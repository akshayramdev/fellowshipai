# Categorize products from image(s) and text descriptors
Based on following paper: https://arxiv.org/pdf/1904.09037.pdf

## Problem Statement:
Relates primarily to products on retail/marketplace sites such as Amazon. The problem deals with 3 main aspects:
* We are trying to categorize products. Number of categories ~5000
* Product has images (1 or more)
* Product has text - title, description, keywords

Possible end-problems to solve: 
a) Find the category, given product images and user provided text description. 
b) Find mis-categorized products

**Please use this file as a _living document_, where key decisions/events are captured.**

## Dataset citations
* R. He, J. McAuley. Modeling the visual evolution of fashion trends with one-class collaborative filtering. WWW, 2016
* J. McAuley, C. Targett, J. Shi, A. van den Hengel. Image-based recommendations on styles and substitutes. SIGIR, 2015

Stored in gs://platform-ai-research/datasets/ProductNet/

## Steps:

1. What are the categories? 
Use Google taxonomy_w_ids file (5400 + categories). 
This is large, make sense to use a subset in the beginning. Go deep or go broad? (5/3/19)

2. Database
Amazon product data

3. Modeling


## Update (5/22/2019): Demo on platform.ai

We selected 10 random classes, with approximately 780 images in total and tried running on platform.ai. The accuracy on the [platform](https://platform.ai/app/imageTraining/projects/ahJzfnBsYXRmb3JtLWFpLXByb2RyUwsSBFVzZXIiHFlpdzFaS3hTNEtRNFM4cEh0Mkt0eUc3ZnZwRTIMCxIHUHJvamVjdCIgYzM5NzE1MGYzNmI0NGUwOGEwNzgxYzc5ZTBlMWUzN2IM) is 78%. Our [latest model](https://github.com/fellowship/platform-demos3/blob/master/ProductNet/workbook_resnet34_images777_classes10.ipynb) managed to achieve an accuracy of 87%. We have been using fast.ai's default with ResNet 34 as the architecture. A more detailed write-up can be found at the [notebook](https://github.com/fellowship/platform-demos3/blob/master/ProductNet/workbook_resnet34_images777_classes10.ipynb).