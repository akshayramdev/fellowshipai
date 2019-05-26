import sys
from pathlib import Path
import sqlite3
import json
import pandas as pd
import os
from PhotoData import PhotoData
import numpy as np
import time
from tqdm import tnrange, tqdm_notebook
import io
import lmdb
from PIL import Image
from IPython.display import display
import time

import skimage.io as io
from fastai.vision import Image as saver
from fastai.vision import Path, pil2tensor
from pycocotools.coco import COCO

import shutil
class Paperdoll(object):
    # Preapre the Paths for the paperdoll database
    # The fork of paperdoll will have the sqlite3 db at: ./data/chictopia
    def __init__(self,down_dir, train_dir):
        print("Directory: ", down_dir)
        self.train_dir = train_dir
        
        self.paperdoll_db_path = os.path.join(down_dir, 'chictopia.sqlite3')
        self.conn_str = 'file:' + str(self.paperdoll_db_path) + '?mode=ro'
    def connect_paperdoll_sql(self):
        # Establish the connection with the paperdoll database and return in a DF
        self.paperdoll_db_conn = sqlite3.connect(self.conn_str, uri=True)
        paperdoll_db_sql = """
            SELECT
                id,
                path
            FROM
                photos
            WHERE
                photos.post_id IS NOT NULL
            AND
                photos.file_file_size IS NOT NULL
        """
        self.df_paperdoll = pd.read_sql(paperdoll_db_sql, con=self.paperdoll_db_conn).astype({
            'id': 'int64',
            'path': 'str',
        })
        
        return self.df_paperdoll

    def get_moda_df(self,down_dir, moda_json, df_paperdoll = None):
        # If you want to use the Moda labels. use this function with the moda .json
        
        
        for modanet_json_path in down_dir.glob(str(moda_json)):
            with open(str(modanet_json_path), 'r', encoding='utf-8') as f:

                modanet_json_data = json.load(f)

                df_modanet = pd.DataFrame.from_dict(modanet_json_data['images']).astype(dtype={
                    'id': 'int64',
                    'file_name': 'str',
                })
            df = df_modanet.merge(df_paperdoll, on='id', how='inner')
            self.moda_df = df
        return df
    
    def load_images_from_sql(self):
        # This function returns all the photos in the sql database
    
        db = sqlite3.connect(self.conn_str, uri=True)
        photos = pd.read_sql("""
            SELECT
                *,
                'http://images2.chictopia.com/' || path AS url
            FROM photos
            WHERE photos.post_id IS NOT NULL AND file_file_size IS NOT NULL
        """, con=db)
        print('Found %d photos' % (len(photos)))
        
        return photos
    
    def save_photos_from_lmdb_to_png(self, moda_df, photos, photo_data):
        # This function matches the images in moda with the corresponding images
        # in the lmdb database and stores it as a png in the train_dir
        # Note, the .id in modaDf doesn't reflect photos.id.
        # Use the following code for better intution after running the for loop
        """
        print(img_id)
        img_id = df.iloc[entry_idx].id
        idx=np.where(photos['id']==img_id)
        print(int(idx[0]))
        photo = photos.iloc[idx]
        df = moda_df
        """
        entry_idx= 0
        for i in tnrange(df.shape[0], desc='Saving images'):
            img_id = df.iloc[entry_idx].id
            idx= int(np.where(photos['id']==img_id)[0])
            photo = photos.iloc[idx]
            img_path = os.path.join(self.train_dir ,"{:07d}.png".format(photo.id))
            if (not photo_data[photo.id] is None) and (not os.path.exists(img_path)):
                photo_data[photo.id].save(img_path,format="PNG")
            entry_idx+=1
        print('Processed: {} entries '.format(entry_idx))
        
        return "Photos saved at: {}".format(self.train_dir)

    
class COCO_(object):

        
    def generate_masks(imgDir,maskDir, annFile, cat_dir, attributes):
        print("changes")
        manifest = dict()
        cat_dir = os.path.join(imgDir,cat_dir)
        print('Saving masks at {}'.format(cat_dir))
        if not os.path.exists(cat_dir):
            os.mkdir(cat_dir)
            print("Creating folder {}".format(cat_dir))
        coco=COCO(annFile)
        # display COCO categories and supercategories
        cats = coco.loadCats(coco.getCatIds())
        print("No of cats: ", len(cats))
        nms=[cat['name'] for cat in cats]
        print('COCO categories: \n{}\n'.format(' '.join(nms)))

        nms = set([cat['supercategory'] for cat in cats])
        print('COCO supercategories: \n{}'.format(' '.join(nms)))
        duplicates = 0
        catIds = coco.getCatIds(catNms=attributes); #['outer','top','headwear']
        imgIds = coco.getImgIds(catIds=catIds );
        imgIds = coco.getImgIds(imgIds = imgIds)
        print("catids:{}, imgids{}".format(len(catIds),len(imgIds)))
        print('Loading {}'.format(imgDir))
        for i in tnrange(len(imgIds), desc='Compiling masks'):
            img = coco.loadImgs(imgIds[i])[0]
            img_fileName = img['file_name'][:-4]+".png" # Removing the jpg extension and replacing with "png"
            dest = os.path.join(imgDir,img_fileName)
            if (os.path.exists(dest)):
                jpg_im = Image.open(dest)
                jpg_im.save(os.path.join(cat_dir,img_fileName))
                
                annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
                anns = coco.loadAnns(annIds)
                mask = coco.annToMask(anns[0])
                for i in range(len(anns)): #Uncomment for combining categories masks..
                    mask = mask | coco.annToMask(anns[i])
                mask_path = os.path.join(maskDir,img_fileName)
                if os.path.exists(mask_path):
                    duplicates+=1
                file_name_path = os.path.join(cat_dir,img_fileName)
                im = Image.fromarray(mask)
                im.save(mask_path)
                manifest[file_name_path] = os.path.join(mask_path)
        print("Done")
        print("Duplicates: ", duplicates)
        return manifest
