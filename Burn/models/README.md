
## Hyperparamters and Model Selection Experiments Summary


* These results have been obtained with num_epoch = 5 and we also observe that the results may change with more num_epoch during training. For instance, resnet-50 becomes the more accurate than resnet-34 models when training with num_epoch = 50 which may not be practical however. 


![Accuracy vs Batch Size](https://github.com/fellowship/platform-demos3/blob/master/Burn/models/bs.png)

Fig.1: Note that optimal batch size is also dpendent on the learning [rate](https://miguel-data-sc.github.io/2017-11-05-first).


![Accuracy vs Image Size](https://github.com/fellowship/platform-demos3/blob/master/Burn/models/im-sz.png)

Fig.2: Note it's common to use sz = 224 as most torch models require sz = 224.


![Accuracy vs Model Architectures](https://github.com/fellowship/platform-demos3/blob/master/Burn/models/models-comparison.png)

Fig.3: Densenet models have the highest accuracy, however, it's known that ResNet models are computationally less expensive while Squeezenet1_1 is known to be more mobile-firendly. VGG models are known to be relatively more resilient for image data with [noise](arxiv.org/abs/1604.04004). Also note ResNet-34 performs well here as the common default parameters of ps=0.5, sz=224 and bs= 64 with num_spoch=5 maybe optimal for ResNet-34 relative to other ResNet models. 
