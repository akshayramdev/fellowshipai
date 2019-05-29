''' 
    Parses the streetstyle manifest file and outputs images and labels
    cropped from the original image using the bounding box/label pairs.
    Some images have multiple bounding boxes/labels so you'll find images
    in the output with similar names, differentiated with a _i appended to
    the end. The output csv contains the mapping between image -> label, 
    you don't need to worry about bounding box

    Adapted from the previous cohorts preprocess code (Jan 2019).
'''

import pandas as pd
from PIL import Image, ImageDraw
from os.path import basename, isdir, isfile, exists
from pathlib import Path

from os import mkdir, environ # just for my use to access environment variables, not required

## edit these for your system
ROOT = Path( "{}/data/streetstyle".format( environ["FASHION_DIR"] ) ) # optional
IMAGES_PATH = ROOT/Path( "streetstyle27k" )
CROPPED_PATH = ROOT/Path( "streetstyle27k_cropped" )
DRAW_PATH = ROOT/Path( "streetstyle27k_drawed" )
MANIFEST_PATH = ROOT/Path( "streetstyle27k.manifest" )
CSV_NAME = ROOT/Path( "ss27k_labels.csv" )
MAP_NAME = ROOT/Path( "ss27k_map.csv" )
IGNORE_NAME = ROOT/Path( "dupe_ignores.csv" )
perform_crop = True # set to false to only output csv (or do neither)
write_csv = True # set to false to only crop (or do neither)
use_ignore = True # ignore bbox IDs from this csv
output_uncropped_img_map = True # use this in case you want to work with the original image (no cropping) - still performs label preprocessing and dupe removal
draw_bboxes = False # output the original image with the bbox drawn in to it (uses same file name as the cropped image but in a different folder)
img_ext = "jpg"
## --------------------------

## make sure our directories exist
if not ( exists( IMAGES_PATH ) and isdir( IMAGES_PATH ) ):
    print( "Can't find the image path!" )
    exit()

if not ( exists( MANIFEST_PATH ) and isfile( MANIFEST_PATH ) ):
    print( "Can't find the manifest file!" )
    exit()

if not ( exists( CROPPED_PATH ) and isdir( CROPPED_PATH ) ):
    mkdir( CROPPED_PATH )

if draw_bboxes and ( not ( exists( DRAW_PATH ) and isdir( DRAW_PATH ) ) ):
    mkdir( DRAW_PATH )
## --------------------------

print( "Reading manifest" )
df = pd.read_csv( MANIFEST_PATH )
print( "Found {} label/bbox pairs".format( len( df ) ) )

label_cols = ['clothing_pattern', 'major_color',
       'wearing_necktie', 'collar_presence', 'wearing_scarf', 'sleeve_length',
       'neckline_shape', 'clothing_category', 'wearing_jacket', 'wearing_hat',
       'wearing_glasses', 'multiple_layers']

for col in df.columns:
    if col in set(label_cols):
        df[col] = (df[col].str.lower()
                    .str.replace(" ", "_")
                    .str.replace("more_than_1_color", "multicolored")
                    .fillna("no_label"))

if use_ignore:
    if not ( exists( IGNORE_NAME ) and isfile( IGNORE_NAME ) ):
        print( "Can't find the image ignore csv!" )
        exit()

    print( "Loading ignore list" )
    idf = set( i[0] for i in pd.read_csv( IGNORE_NAME )[["id"]].to_numpy( dtype=int ) )

print( "Processing image bboxes" )
images = {}
images_map = {}
total = len( df )
for n, ( _, item ) in enumerate( df[["id","url", "x1", "x2", "y1", "y2"] + label_cols].iterrows() ):
    if use_ignore and item["id"] in idf:
        continue

    uri = item["url"]
    fn = basename(uri)
    fn_split = fn.split( "." )
    fn_no_ext, fn_ext = ".".join(fn_split[:-1]), img_ext
    img_path_local = "{}/{}/{}/{}".format( fn[0], fn[1], fn[2], fn )
    img_path = IMAGES_PATH/img_path_local

    cropped_img_name = "{}_{}.{}".format( fn_no_ext, item["id"], fn_ext )
    
    bbox = item["x1"], item["y1"], item["x2"], item["y2"]

    if perform_crop and not exists( CROPPED_PATH/cropped_img_name ):
        im_crop = Image.open( img_path ).crop( bbox )
        im_crop.save( CROPPED_PATH/cropped_img_name )

    if draw_bboxes and not exists( DRAW_PATH/cropped_img_name ):
        im = Image.open( img_path )
        im_bbox = ImageDraw.Draw( im )
        im_bbox.rectangle( [ item["x1"], item["y1"], item["x2"], item["y2"] ], outline="red", width=10 )
        im.save( DRAW_PATH/cropped_img_name )

    labels = " ".join( [ "{}_{}".format( k, item[k] ) for k in label_cols ] )
    images[cropped_img_name] = labels

    if output_uncropped_img_map:
        images_map[item["id"]] = [ img_path_local, cropped_img_name, str( item["x1"] ), 
                str( item["y1"] ), str( item["x2"] ), str( item["y2"] ), labels ]
    
    if n%1000 == 0:
        print( "{}% done".format( round( ( n / total ) * 100.0 ) ) )

drop_string = "dropping {} in the process".format( len( idf ) ) if use_ignore else ""
print( "Processed {} original images in to {} crops {}".format( len( df["url"].unique() ), len( images ), drop_string ) )

if write_csv:
    print( "Writing csv" )
    with open( CSV_NAME, "w" ) as f:
        f.write( "image,label\n" )
        for k, v in images.items():
            f.write( "{},{}\n".format( k, v ) )

if output_uncropped_img_map:
    print( "Writing mapping data" )
    with open( MAP_NAME, "w" ) as f:
        f.write( "id,old_image,image,x1,y1,x2,y2,label\n" )
        for k, v in images_map.items():
            f.write( "{},{}\n".format( k, ",".join( v ) ) )

print( "Done!" )