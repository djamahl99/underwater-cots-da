import numpy as np
import matplotlib.pyplot as plt
import sys
from google.protobuf import json_format
import json
import cv2
import os
from pathlib import Path 
from collections import defaultdict
from tqdm import tqdm
import torchvision
from PIL import Image
import shutil
from dataset.kaggle_aims_pair_dataset_w_boxes import kaggle_aims_pair_boxed

if __name__ == "__main__":
    new_size = (1080,1920)

    split = sys.argv[1]

    # ds = kaggle_aims_pair_boxed(aims_split=split)
    # new_size = ds.size

    resize = torchvision.transforms.Compose(transforms=[
        torchvision.transforms.Resize(size=new_size),
        torchvision.transforms.ToTensor()
    ]) # original is 5312x3040 (not exactly 16:9 but just resizing to)


    path = "/home/etc004/code/mmyolo/data/aims"
    annot_dir = f"{path}"

    new_path = "../AIMS_data_test"
    new_annot_dir = f"{new_path}/annotations"

    if not os.path.exists(new_annot_dir):
        os.makedirs(new_annot_dir)

    data = {}


    shutil.copyfile(f"{annot_dir}/{split}", f"{new_annot_dir}/{split}")

    with open(f"{annot_dir}/{split}", 'r') as f:
        data = json.load(f) 

    imgToAnns = {x['id']: [] for x in data['images']}
    
    ann_id = 0
    for x in data['annotations']:
        x['segmentation'] = None
        x['id'] = ann_id
        ann_id += 1
        imgToAnns[x['image_id']].append(x)

    with tqdm(desc=f"Resizing", unit='image') as pbar:

        # copy images and resize    
        for x in tqdm(data['images']):
            old_filename = f"{path}/{x['file_name']}"
            new_filename = f"{new_path}/{x['file_name']}"

            old_size = (x['height'], x['width'])

            ratios = (new_size[0] / x['height'], new_size[1] / x['width'])
            print("ratios", ratios)
            print("anns", imgToAnns[x['id']])

            anns_ = []
            for ann in imgToAnns[x['id']]:
                print("ann", ann)
                box = np.array(ann['bbox'])
            
                box[0::2] *= ratios[1] # x *= r_width
                box[1::2] *= ratios[0] # y *= r_height

                area = int(box[2] * box[3])

                print("box diff", ann['bbox'], box)
                ann['bbox'] = list(box)
                ann['area'] = area

                anns_.append(ann)
            
            imgToAnns[x['id']] = anns_
                

            # exit()

            if os.path.exists(new_filename):
                continue

            image = Image.open(old_filename)
            image = resize(image)


            if not os.path.exists(Path(new_filename).parent):
                os.makedirs(str(Path(new_filename).parent))

            # print(f"{path}/{x['file_name']}", new_filename)
            pbar.update(1)

            p_ = Path(new_filename)

            # pbar.set_postfix(**{"parent": p_.parents[0].replace(p_.parents[1], ""), "gparent": p_.parents[1].replace(p_.parents[2], "")})

            torchvision.utils.save_image(image, new_filename)
            # shutil.copyfile(old_filename, new_filename)
    
    for i in range(len(data['images'])):
        data['images'][i]['height'], data['images'][i]['width'] = new_size
        
    data['annotations'] = []
    for x in imgToAnns.values():
        data['annotations'].extend(x)

    with open(f"{new_annot_dir}/{split}", "w") as f:
        json.dump(data, f)