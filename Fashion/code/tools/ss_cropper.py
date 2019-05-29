import pandas as pd
from PIL import Image
from os.path import isdir, isfile, exists
from pathlib import Path

from os import mkdir, environ # just for my use to access environment variables, not required

## edit these for your system
ROOT = Path( "{}/data/streetstyle".format( environ["FASHION_DIR"] ) ) # optional
IMAGES_PATH = ROOT/Path( "streetstyle27k" )
CROPPED_PATH = ROOT/Path( "streetstyle27k_imageai_cropped" )
CSV_IN = ROOT/Path( "ss27k_imageai_bbox.csv" )
CSV_OUT = ROOT/Path( "ss27k_imageai_labels.csv" )
perform_crop = True # set to false to only output csv (or do neither)
write_csv = True # set to false to only crop (or do neither)
img_ext = "jpg"
## --------------------------

## make sure our directories exist
if not ( exists( IMAGES_PATH ) and isdir( IMAGES_PATH ) ):
    print( "Can't find the image path!" )
    exit()

if not ( exists( CROPPED_PATH ) and isdir( CROPPED_PATH ) ):
    mkdir( CROPPED_PATH )
## --------------------------

print( "Reading csv" )
df = pd.read_csv( CSV_IN )
print( "Found {} label/bbox pairs".format( len( df ) ) )

print( "Processing image bboxes" )
images = {}
total = len( df )
for n, ( _, item ) in enumerate( df[["old_image", "image", "x1", "x2", "y1", "y2", "labels"]].iterrows() ):
    bbox = item["x1"], item["y1"], item["x2"], item["y2"]

    if perform_crop and not exists( CROPPED_PATH/item["image"] ):
        im_crop = Image.open( IMAGES_PATH/item["old_image"] ).crop( bbox )
        im_crop.save( CROPPED_PATH/item["image"] )

    images[item["image"]] = item["labels"]
    
    if n%1000 == 0:
        print( "{}% done".format( round( ( n / total ) * 100.0 ) ) )

if write_csv:
    print( "Writing csv" )
    with open( CSV_OUT, "w" ) as f:
        f.write( "image,label\n" )
        for k, v in images.items():
            f.write( "{},{}\n".format( k, v ) )

print( "Done!" )