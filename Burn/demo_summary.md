# Burn depth classification

Burns cause an estimated 180,000 deaths
per year, primarily in the developing world
[(1)](https://www.who.int/news-room/fact-sheets/detail/burns).
While the rate of burn-related deaths is decreasing in many
high-income countries, in the United States, approximately 486,000
people still sought medical treatment for burn injuries in 2016
[(2)](https://ameriburn.org/who-we-are/media/burn-incidence-fact-sheet/).
The prolonged hospitalization, disability, and disfigurement that commonly
accompany non-fatal burn injuries are leading causes of morbidity in
regions where burn incidence remains high.

Correctly identifying the depth of a burn is a critical step towards
treating it, with the primary method of burn classification being visual
inspection.  However, standard clinical methods are only 60-80% accurate
[(3)](https://www.sciencedirect.com/science/article/abs/pii/S0305417900001182?via%3Dihub).
To improve outcomes for burn patients, we aim to train a neural network
to classify burn depth based on standard images, such as those taken by
a smartphone camera.  The eventual goal is a smartphone or web app that
can be used in situations where a burn expert is unavailable, such as
low-resource environments or by first responders in crisis situations.
Working in collaboration with a burn specialist, we anticipate having
access to a sizable collection of high-quality, expertly-labeled training
data.  As a preliminary step, we are using a web-scraped dataset of 742
images of unburned skin and first, second, and third degree burns.

Using platform.ai, we used these images to train an image
[classifier](https://platform.ai/app/imageTraining/projects/ahJzfnBsYXRmb3JtLWFpLXByb2RyUwsSBFVzZXIiHEZUSXdNOThIazVlT2Q0ajJnWFpQRGV5WG42dTEMCxIHUHJvamVjdCIgMzIxZDVhNzA3OGExNDNiOWFkYmY1NmJiYTI2NDc5NWIM)
to over 80% accuracy, measured on a randomly-selected hold out
validation set.  In tandem, we also trained several image classifiers
using the fastai library.  Comparing several different architectures,
we found that the one that gave the best overall accuracy on our dataset
was DenseNet-161.  In addition to the default set of data augmentation
functions suggested by fastai, we also included vertical flips of
the images, and added [mixup](https://arxiv.org/abs/1710.09412).
Combined with test-time augmentation, our fastai
[classifier](https://github.com/fellowship/platform-demos3/blob/master/Burn/burn2.ipynb)
reached >94% accuracy.

Because both our models rely on image labels from non-burn experts
(i.e. us, or a previous group of machine learning fellows), we can
reasonably expect our labeling accuracy to be less than 80%, implying
that the human labelers in this training pipeline are a major source of
inaccuracy in the resulting model.  However, as medical professionals
who are not burn experts classify burn depth correctly only 60-80% of the
time, our result hints at the potential for a significant improvement over
standard practices, even within medical settings.  A future model trained
on expertly-labeled images may well surpass human-level performance on
burn identification.

