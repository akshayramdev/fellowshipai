'''
    Code adapted from https://github.com/floodsung/LearningToCompare_FSL for the paper
    https://arxiv.org/abs/1711.06025.

    #-------------------------------------
    # Project: Learning to Compare: Relation Network for Few-Shot Learning
    # Date: 2017.9.21
    # Author: Flood Sung
    # All Rights Reserved
    #-------------------------------------
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import math
from pathlib import Path

from .data import RNDataBuilder, RNImageStream



class RNEncoder(nn.Module):
    """docstring for RNEncoder"""
    def __init__(self,feature_size,num_channels=3):
        super(RNEncoder, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(num_channels,feature_size,kernel_size=3,padding=0),
                        nn.BatchNorm2d(feature_size, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(feature_size,feature_size,kernel_size=3,padding=0),
                        nn.BatchNorm2d(feature_size, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(feature_size,feature_size,kernel_size=3,padding=1),
                        nn.BatchNorm2d(feature_size, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(feature_size,feature_size,kernel_size=3,padding=1),
                        nn.BatchNorm2d(feature_size, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class RNRelation(nn.Module):
    """docstring for RNRelation"""
    def __init__(self,feature_size,hidden_size,num_channels=3,padding=0):
        super(RNRelation, self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(feature_size*2,feature_size,kernel_size=3,padding=padding),
                        nn.BatchNorm2d(feature_size, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(feature_size,feature_size,kernel_size=3,padding=padding),
                        nn.BatchNorm2d(feature_size, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.fc1 = nn.Linear(feature_size*num_channels*num_channels,hidden_size)
        self.fc2 = nn.Linear(hidden_size,1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0),-1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out


class RelationNetwork():
    def __init__( self, df=None, data_dir=None, model_dir=None, val_df=None, test_df=None, valid_pcnt=0.2, 
            test_pcnt=0, data_num_dims=3, feature_dim=64, relation_dim=8, class_num=5, 
            num_per_class=5, num_query=15, episodes=100000, val_episodes=1000, 
            learning_rate=0.001, hidden_unit=10, gpu=0, shuffle=False, padding=0 ):

        # dataset variables
        self.df = df
        self.data_dir = data_dir
        self.model_dir = model_dir if not None else self.data_dir
        self.val_df = val_df
        self.val_pcnt = valid_pcnt
        self.test_df = test_df
        self.test_pcnt = test_pcnt
        self.class_num = class_num
        self.num_per_class = num_per_class
        self.num_query = num_query
        self.shuffle = shuffle

        # model variables
        self.gpu = gpu
        self.episodes = episodes
        self.val_episodes = val_episodes
        self.feature_dim = feature_dim

        # build the model
        self.feature_encoder = RNEncoder(feature_dim,data_num_dims)
        self.relation_network = RNRelation(feature_dim,relation_dim,data_num_dims,padding)

        self.feature_encoder.apply(self._weights_init)
        self.relation_network.apply(self._weights_init)

        self.feature_encoder.cuda(self.gpu)
        self.relation_network.cuda(self.gpu)

        self.feature_encoder_optim = torch.optim.Adam(self.feature_encoder.parameters(),lr=learning_rate)
        self.feature_encoder_scheduler = StepLR(self.feature_encoder_optim,step_size=100000,gamma=0.5)
        self.relation_network_optim = torch.optim.Adam(self.relation_network.parameters(),lr=learning_rate)
        self.relation_network_scheduler = StepLR(self.relation_network_optim,step_size=100000,gamma=0.5)

        # store max test accuracy
        self.max_acc = 0

    def _build_dataset( self ):
        raise NotImplementedError()

    def train( self, save_improvement=True, test_on=5000, print_on=100 ):
        print( "Beginning train" )
        for ep in range( self.episodes ):
            self.feature_encoder_scheduler.step(ep)
            self.relation_network_scheduler.step(ep)

            self.dataset.resample_train()
            
            s_dataloader = self.sample_datastream.make_dataloader( self.dataset.sample )
            q_dataloader = self.query_datastream.make_dataloader( self.dataset.query )

            samples,sample_labels = s_dataloader.__iter__().next()
            batches,batch_labels = q_dataloader.__iter__().next()

            sample_features = self.feature_encoder(Variable(samples).cuda(0))
            sample_shape = sample_features.shape
            sample_features = sample_features.view(self.class_num,self.num_per_class,self.feature_dim,sample_shape[-1],sample_shape[-2])
            sample_features = torch.sum(sample_features,1).squeeze(1)
            batch_features = self.feature_encoder(Variable(batches).cuda(0))

            sample_features_ext = sample_features.unsqueeze(0).repeat(self.num_query*self.class_num,1,1,1,1)
            batch_features_ext = batch_features.unsqueeze(0).repeat(self.class_num,1,1,1,1)
            batch_features_ext = torch.transpose(batch_features_ext,0,1)
            sample_ext_shape = sample_features_ext.shape
            relation_pairs = torch.cat((sample_features_ext,batch_features_ext),2).view(-1,self.feature_dim*2,19,19)
            relations = self.relation_network(relation_pairs).view(-1,self.class_num)

            mse = nn.MSELoss().cuda(0)
            one_hot_labels = Variable(torch.zeros(self.num_query*self.class_num, self.class_num).scatter_(1, batch_labels.view(-1,1), 1).cuda(0))
            loss = mse(relations,one_hot_labels)

            self.feature_encoder.zero_grad()
            self.relation_network.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.feature_encoder.parameters(),0.5)
            torch.nn.utils.clip_grad_norm_(self.relation_network.parameters(),0.5)

            self.feature_encoder_optim.step()
            self.relation_network_optim.step()

            if type( print_on ) is int and ( ep+1 )%print_on == 0:
                print( "Episode:", ep+1, "Loss:", loss.item() )

            if type( test_on ) is int and ep%test_on == 0:
                self.validate( save_improvement )

    def validate( self, save_improvement=True ):
        print( "Scoring on validation set..." )
        acc = 0
        for _ in range( self.val_episodes ):
            self.dataset.resample_val()

            s_dataloader = self.support_datastream.make_dataloader( self.dataset.val_support )
            q_dataloader = self.test_datastream.make_dataloader( self.dataset.val_test )

            sample_images,sample_labels = s_dataloader.__iter__().next()
            test_images,test_labels = q_dataloader.__iter__().next()
            test_labels = test_labels.cuda( self.gpu )
            
            batch_size = test_labels.shape[0]

            sample_features = self.feature_encoder(Variable(sample_images).cuda( self.gpu ))
            sample_shape = sample_features.shape
            sample_features = sample_features.view(self.class_num,self.num_per_class,self.feature_dim,sample_shape[-1],sample_shape[-2])
            sample_features = torch.sum(sample_features,1).squeeze(1)
            test_features = self.feature_encoder(Variable(test_images).cuda(self.gpu))

            sample_features_ext = sample_features.unsqueeze(0).repeat(batch_size,1,1,1,1)

            test_features_ext = test_features.unsqueeze(0).repeat(1*self.class_num,1,1,1,1)
            test_features_ext = torch.transpose(test_features_ext,0,1)
            relation_pairs = torch.cat((sample_features_ext,test_features_ext),2).view(-1,self.feature_dim*2,19,19)
            relations = self.relation_network(relation_pairs).view(-1,self.class_num)

            _,predict_labels = torch.max(relations.data,1)

            rewards = [1 if predict_labels[j]==test_labels[j] else 0 for j in range(self.class_num*self.num_query)]

            acc += np.sum(rewards)

        acc = acc/1.0/self.class_num/self.num_query/self.val_episodes
        print( "Validation set accuracy:", acc )

        if save_improvement and ( acc > self.max_acc ):
            self.save()

        self.max_acc = max( acc, self.max_acc )

    def save( self, dir=None ): 
        print( "Saving model" )
        feature_name = "feature_encoder_{}_way_{}_shot.pkl".format( self.class_num, self.num_per_class )
        relation_name = "relation_network_{}_way_{}_shot.pkl".format( self.class_num, self.num_per_class )
        if dir is None:
            feature_name = self.model_dir/"models"/feature_name
            relation_name = self.model_dir/"models"/relation_name
        else:
            feature_name = dir/feature_name
            relation_name = dir/relation_name

        torch.save( self.feature_encoder.state_dict(), feature_name )
        torch.save( self.relation_network.state_dict(), relation_name )

    def load( self, dir=None ):
        print( "Loading model" )
        feature_name = "feature_encoder_{}_way_{}_shot.pkl".format( self.class_num, self.num_per_class )
        relation_name = "relation_network_{}_way_{}_shot.pkl".format( self.class_num, self.num_per_class )
        if dir is None:
            feature_name = self.model_dir/"models"/feature_name
            relation_name = self.model_dir/"models"/relation_name
        else:
            feature_name = dir/feature_name
            relation_name = dir/relation_name

        self.feature_encoder.load_state_dict( torch.load( feature_name ) )
        self.relation_network.load_state_dict( torch.load( relation_name ) )

    def _weights_init( self, m ):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif classname.find('Linear') != -1:
            n = m.weight.size(1)
            m.weight.data.normal_(0, 0.01)
            m.bias.data = torch.ones(m.bias.data.size())


class ImageRelationNetwork( RelationNetwork ):
    def __init__( self, *args, tsf=None, size=224, **kwargs ):
        super().__init__( *args, **kwargs )

        self._build_dataset( tsf=tsf, size=size )
    
    def _build_dataset( self, tsf=None, size=None ):
        self.dataset = RNDataBuilder( self.df, self.class_num, self.num_query, self.num_per_class, val_df=self.val_df, val_pcnt=self.val_pcnt, test_df=self.test_df, test_pcnt=self.test_pcnt )
    
        self.sample_datastream = RNImageStream( self.data_dir, self.class_num, self.num_per_class, tsf=tsf, size=size  )
        self.query_datastream = RNImageStream( self.data_dir, self.class_num, self.num_query, shuffle=self.shuffle, tsf=tsf, size=size )

        self.support_datastream = RNImageStream( self.data_dir, self.class_num, self.num_per_class, tsf=tsf, size=size )
        self.test_datastream = RNImageStream( self.data_dir, self.class_num, self.num_query, shuffle=self.shuffle, tsf=tsf, size=size )




if __name__ == '__main__':
    print( "Loading dataset" )
    data_dir = Path( "/home/lewis/Work/Employment/fellowshipai/fashion/data/omniglot" )
    df = pd.read_csv( data_dir/"data.csv", sep="," )

    print( df.head() )

    norm = transforms.Normalize( mean=[0.92206], std=[0.08426] )
    rot = transforms.RandomRotation( [0, 364] )
    tsf = transforms.Compose( [rot, transforms.ToTensor(), norm] )

    print( "Built transforms" )
    
    rn = ImageRelationNetwork( df=df, data_dir=data_dir, valid_pcnt=0.2, data_num_dims=1, padding=1, shuffle=True, tsf=tsf, size=28 )

    print( "Loaded architecture" )
    rn.train()