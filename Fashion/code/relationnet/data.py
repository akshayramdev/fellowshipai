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


import pandas as pd
import numpy as np

from pathlib import Path
from PIL import Image
from random import sample, shuffle, choice

import torchvision.transforms as transforms
from torch import randperm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from functools import lru_cache

class RNDataBuilder():
    """
        Generates sample/query and support/test sets from a pandas dataframe in the format 
        the relational network paper (https://arxiv.org/pdf/1711.06025.pdf) expects.
    """

    def __init__( self, train_df, num_support_class, num_query, num_support, val_df=None, val_pcnt=0.1, test_df=None, test_pcnt=0.1, obj_idx=0, label_idx=1 ):
        self.C = num_support_class
        self.K = num_support
        self.N = num_query

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        #train_df = train_df[[obj_idx, label_idx]]
        self.train_classes = train_df["label"].unique()
        class_len = len( self.train_classes )

        if val_df is not None:
            #val_df = val_df[[obj_idx, label_idx]]
            self.val_classes = list( val_df["label"].unique() )

            if len( self.val_classes ) < self.C:
                raise IndexError( "The dataset does not have enough classes to meet the validation set requirements." )
        elif val_pcnt > 0:
            self.val_classes = list( np.random.choice( self.train_classes, size=int( class_len*val_pcnt ), replace=False ) )
            self.val_df = train_df
        else:
            self.val_classes = []   
        
        if test_df is not None:
            #test_df = test_df[[obj_idx, label_idx]]
            self.test_classes = list( test_df["label"].unique() )

            if len( self.test_classes ) < self.C:
                raise IndexError( "The dataset does not have enough classes to meet the test set requirements." )
        elif test_pcnt > 0:
            self.test_classes = list( np.random.choice( self.train_classes, size=int( class_len*test_pcnt ), replace=False ) )
            self.test_df = train_df
        else:
            self.test_classes = []

        if len( self.train_classes ) < self.C:
            raise IndexError( "The dataset does not have enough classes to meet the training set requirements." )

        self.train_classes = list( self.train_classes )

    @lru_cache( maxsize=None )
    def _get_class_examples( self, clas, mode ):
        if mode == "train":
            df = self.train_df
        elif mode == "valid":
            df = self.val_df
        elif mode == "test":
            df = self.test_df
        else:
            raise ValueError( "Mode must be 'train', 'valid' or 'test'.")

        return df.loc[df["label"] == clas]

    def _sample_from_classes( self, class_list, mode ):
        support_length, query_length = len( class_list ) * self.K, len( class_list ) * self.N
        support_data = np.empty( shape=(support_length,2), dtype='U100' )
        query_data = np.empty( shape=(query_length,2), dtype='U100' )
        
        class_id = dict( zip( class_list, range( len( class_list ) ) ) )
        for n, clas in enumerate(class_list):
            data_sample = self._get_class_examples( clas, mode ).sample( self.K+self.N )
            data_sample["label"] = data_sample["label"].map( class_id )
            data_sample = data_sample.to_numpy( dtype=str )

            support_data[n*self.K:(n*self.K)+self.K,:] = data_sample[:self.K]
            query_data[n*self.N:(n*self.N)+self.N,:] = data_sample[self.K:self.K+self.N]

        return support_data, query_data

    def resample_val( self ):
        sample_classes = sample( self.val_classes, self.C )
        self.val_support, self.val_test = self._sample_from_classes( sample_classes, "valid" )

    def resample_test( self ):
        self.test_support, self.test_test = self._sample_from_classes( self.test_classes, "test" )

    def resample_train( self ):
        sample_classes = sample( self.train_classes, self.C )
        self.sample, self.query = self._sample_from_classes( sample_classes, "train" )


class RNDataStream( Dataset ):
    def __init__( self, data_dir, num_classes, num_examples, shuffle=False ):
        self.root = data_dir
        self.data = []
        self.shuffle = shuffle

        self.n_clas = num_classes
        self.n_eg = num_examples

    def __len__( self ):
        return len( self.data )

    def __getitem__( self, idx ):
        raise NotImplementedError( "This is an abstract class. Subclass this class for your particular dataset." )

    def make_dataloader( self, data ):
        self.data = data

        sampler = RNDataSampler( self.n_clas, self.n_eg, shuffle=self.shuffle )
        return DataLoader( self, batch_size=self.n_clas*self.n_eg, sampler=sampler )


class RNImageStream( RNDataStream ):
    def __init__( self, *args, tsf=None, size=224, convert="RGB", **kwargs ):
        super().__init__( *args, **kwargs )
        self.transform = tsf
        self.size = size
        self.convert = convert

    def __getitem__( self, idx ):
        image = Image.open( self.root/self.data[idx][0] )
        image = image.convert( self.convert )
        image = image.resize( ( self.size, self.size ), resample=Image.LANCZOS )
        label = self.data[idx][1]

        try:
            label = int(label)
        except ValueError:
            pass

        if self.transform is not None:
            image = self.transform( image )

        return image, label

class RNDataSampler( Sampler ):
    def __init__( self, num_classes, num_examples, shuffle=False ):
        self.n_clas = num_classes
        self.n_eg = num_examples
        self.shuffle = shuffle

    def __iter__( self ):
        if self.shuffle:
            batch = [ [ i+j*self.n_eg for i in randperm( self.n_eg ) ] for j in range( self.n_clas ) ]
        else:
            batch = [ [ i+j*self.n_eg for i in range( self.n_eg ) ] for j in range( self.n_clas ) ]
        batch = [ item for sublist in batch for item in sublist ]

        if self.shuffle:
            shuffle( batch )

        return iter( batch )

    def __len__( self ):
        return 1