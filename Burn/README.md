# Burn classification

## Current progress
Accuracy 0.9459 with DenseNet-161 and TTA (scale=1.05), see end of burn1.ipynb


## Hyperparamters and model slection summary
- from burn_nosplit_data_hyperparameters.ipynb

In summary: optimal values of hyperparameters

- lr = slice(1e-3,1e-2), lr_max= 0.1
- top losses from 1/2, 2/3, and mostly (1-3)/none classes
- bs = 128 or 224
- ps = 0.2
- arch = ResNet50

![Confusion Matrix ResNet34](https://github.com/fellowship/platform-demos3/blob/master/Burn/conf_matrix.png)

![Accuracy vs Droupout rates](https://github.com/fellowship/platform-demos3/blob/master/Burn/ps.png)

![Accuracy vs Model Architectures](https://github.com/fellowship/platform-demos3/blob/master/Burn/arch.png)

