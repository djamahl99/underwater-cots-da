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

if __name__ == "__main__":
    resize = torchvision.transforms.Compose(transforms=[
        torchvision.transforms.Resize(size=(1080,1920)),
        torchvision.transforms.ToTensor()
    ]) # original is 5312x3040 (not exactly 16:9 but just resizing to)


    path = "/home/etc004/code/mmyolo/data/aims"
    annot_dir = f"{path}"

    new_path = "AIMS_data_test"
    new_annot_dir = f"{new_path}/annotations"

    if not os.path.exists(new_annot_dir):
        os.makedirs(new_annot_dir)

    split = "train.json"
    data = {}

    shutil.copyfile(f"{annot_dir}/{split}", f"{new_annot_dir}/{split}")

    with open(f"{annot_dir}/{split}", 'r') as f:
        data = json.load(f) 

    with tqdm(desc=f"Resizing", unit='image') as pbar:

        # copy images and resize    
        for x in tqdm(data['images']):
            old_filename = f"{path}/{x['file_name']}"
            image = Image.open(old_filename)
            image = resize(image)


            new_filename = f"{new_path}/{x['file_name']}"
            if not os.path.exists(Path(new_filename).parent):
                os.makedirs(str(Path(new_filename).parent))

            # print(f"{path}/{x['file_name']}", new_filename)
            pbar.update(1)

            p_ = Path(new_filename)

            # pbar.set_postfix(**{"parent": p_.parents[0].replace(p_.parents[1], ""), "gparent": p_.parents[1].replace(p_.parents[2], "")})

            torchvision.utils.save_image(image, new_filename)
            # shutil.copyfile(old_filename, new_filename)