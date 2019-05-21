import pandas as pd
import os
import cv2
from PIL import Image
from ImageAI.imageai.Detection import ObjectDetection
from collections import defaultdict


execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

df1 = pd.read_csv('StreetStyle.csv')
df1 = df1.drop('Unnamed: 0',axis=1)
df_dict = defaultdict(list)
base_path = "highres-full/"
curr_path = "streetstyle27k/"


def crop_streetstyle(image_path, img_coords, saved_location):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """

    img1 = cv2.imread(image_path)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    blurr = cv2.Laplacian(gray, cv2.CV_64F).var()
    if int(blurr) > 150:
        image_obj = Image.open(image_path)
        cropped_image = image_obj.crop(img_coords)
        cropped_image.save(saved_location)
        return True
    else:
        return False


def opt_person_coord(element):
    coord_list = []
    opt_coords = None
    for x in element[1]:
        if x['name'] == 'person':
            coord_list.append((x['box_points']))

    if len(coord_list) == 1:
        opt_coords = tuple(coord_list[0])

    return opt_coords

def find_name(x):
    t1 = x.split('/')[-1]
    t1 = base_path + t1
    return t1

df1['img_names'] = df1['img_paths']
df1['img_names'] = df1['img_names'].apply(find_name)


for val in range(len(df1)):
    curr_img_path = curr_path + str(df1.iloc[val,23])
    crop_img_path = df1.iloc[val,25]

    detected = detector.detectObjectsFromImage(input_image=os.path.join(execution_path,
                                                                           curr_img_path),
                                                  extract_detected_objects=False, output_type='array')
    try:
        person_coord = opt_person_coord(detected)
        if person_coord:
            x = crop_streetstyle(curr_img_path, person_coord, crop_img_path)
            if x:
                df_dict['img_paths'].append(df1.iloc[val, 23])
                df_dict['crop_path'].append(crop_img_path)
        else:
            pass
    except:
        pass

single_img_df = pd.DataFrame.from_dict(df_dict)
single_img_df.to_csv('highres-full.csv', columns=single_img_df.columns)
