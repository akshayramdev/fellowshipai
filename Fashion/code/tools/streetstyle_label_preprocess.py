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
from PIL import Image
from os.path import basename
from pathlib import Path

from os import environ # just for my use to access environment variables, not required

## edit these for your system
ROOT = Path( "{}/data/streetstyle".format( environ["FASHION_DIR"] ) ) # optional
IMAGES_PATH = ROOT/Path( "streetstyle27k" )
CROPPED_PATH = ROOT/Path( "streetstyle27k_cropped" )
MANIFEST_PATH = ROOT/Path( "streetstyle27k.manifest" )
CSV_NAME = ROOT/Path( "ss27k_labels.csv" )
perform_crop = True # set to false to only output csv (or do neither)
write_csv = True # set to false to only crop (or do neither)
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

print( "Processing image bboxes" )
images = {}
for _, item in df[["url", "x1", "x2", "y1", "y2"] + label_cols].iterrows():
    uri = item["url"]
    fn = basename(uri)
    fn_split = fn.split( "." )
    fn_no_ext, fn_ext = ".".join(fn_split[:-1]), fn_split[-1]
    img_path = IMAGES_PATH/fn[0]/fn[1]/fn[2]/fn

    i = 0
    cropped_img_name = "{}_{}.{}".format( fn_no_ext, i, fn_ext )
    while cropped_img_name in images:
        i += 1
        cropped_img_name = "{}_{}.{}".format( fn_no_ext, i, fn_ext )
        
    if perform_crop:
        bbox = item["x1"], item["y1"], item["x2"], item["y2"]
        im = Image.open(img_path).crop(bbox)
        im.save(CROPPED_PATH/cropped_img_name)

    images[cropped_img_name] =  [ "{}_{}".format( k, item[k] ) for k in label_cols ]
print( "Processed {} original images in to {} crops".format( len( df["url"].unique() ), len( images ) ) )

if write_csv:
    print( "Writing csv" )
    with open( CSV_NAME, "w" ) as f:
        f.write( "image,label\n" )
        for k, v in images.items():
            f.write( "{},{}\n".format( k, " ".join( v ) ) )

print( "Done!" )