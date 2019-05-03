Categorize products from image(s) and text descriptors Based on following paper: https://arxiv.org/pdf/1904.09037.pdf

Problem Statement: Relates primarily to products on retail/marketplace sites such as Amazon. The problem deals with 3 main aspects:

We are trying to categorize products. Number of categories ~5000
* Product has images (1 or more)
* Product has text - title, description, keywords
* Possible end-problems to solve: a) Find the category, given product images and user provided text description. b) Find mis-categorized products

Please use this file as a living document, where key decisions/events are captured.

Steps:
1. What are the categories? 
Use Google taxonomy_w_ids file (5400 + categories). 
This is large, make sense to use a subset in the beginning. Go deep or go broad? (5/3/19)

2. Need a starting database
* Ask Cdiscount (Kaggle competition)
* Scrape online retailers websites - must have description and images.

3. Modeling
