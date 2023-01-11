import torch
from torch.utils import data
import json
from pathlib import Path
import os.path as osp

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from tqdm import tqdm
import numpy as np

import torchvision.transforms as T

class kaggle_aims_pair(data.Dataset):
    def __init__(self) -> None:
        super().__init__()

        size = (288, 512)
        size = (768, 1280)

        # transformer
        # size = (256, 256)
        
        # add more later
        self.transform_kaggle = T.Compose(transforms=[
            #T.CenterCrop(1080),
            T.Resize(size),
            # T.RandomPerspective(distortion_scale=0.1),
            # T.RandomAutocontrast(p=0.1),
            # T.RandomEqualize(),
            # T.RandomAdjustSharpness(),
            # T.RandomRotation(degrees=1),
            # T.AugMix(),
            T.ToTensor()
        ])

        self.transform_aims = T.Compose(transforms=[
            #T.CenterCrop(3040),
            T.Resize(size),
            T.ToTensor()
        ])

        self.kaggle_root = "/home/etc004/code/YOLOX/data/Kaggle_1080_google_v1"
        # self.aims_root = "/home/etc004/code/YOLOX/AIMS_data_test"
        # self.aims_root = "/mnt/d61-visage-data/work/datasets/"
        self.aims_root = "AIMS_data_test"

        # just use train :)
        self.kaggle_anns = str(Path(self.kaggle_root) / "annotations/mmdet_split_train.json")
        self.aims_anns = str(Path(self.aims_root) / "annotations/train.json")
        # self.aims_anns = "aims_sep22.json"

        with open(self.kaggle_anns, 'r') as f:
            kaggle_data = json.load(f)

        with open(self.aims_anns, 'r') as f:
            aims_data = json.load(f)

        # fix aims location changes
        new_imgs = []
        for k in aims_data['images']:
            img = k
            # img['file_name'] = img['file_name'].replace("data/", "")
            new_imgs.append(img)

        aims_data['images'] = new_imgs

        kaggle_imgs_labels = {}
        aims_imgs_labels = {}

        cots_cat_id_kaggle = 0
        cots_cat_id_aims = 1

        print("cats", aims_data['categories'], kaggle_data['categories'])
        
        for ann in tqdm(aims_data['annotations']):
            if ann['category_id'] == cots_cat_id_aims:
                aims_imgs_labels[ann['image_id']] = 1.0

        for ann in tqdm(kaggle_data['annotations']):
            if ann['category_id'] == cots_cat_id_kaggle:
                kaggle_imgs_labels[ann['image_id']] = 1.0

        ms = -1

        # self.aims_imgs_list = [self.aims_root + "/" + x['file_name'] for x in aims_data['images']]
        self.aims_imgs_list = [{
            'image_id': x['id'], 
            'file_name': self.aims_root + "/" + x['file_name'],
            'label': 0.0 if x['id'] not in aims_imgs_labels else 1.0
            } for x in tqdm(aims_data['images'][0:ms])]

        # self.kaggle_imgs_list = [self.kaggle_root + "/" + x['file_name'] for x in kaggle_data['images']]
        self.kaggle_imgs_list = [{
            'image_id': x['id'], 
            'file_name': self.kaggle_root + "/" + x['file_name'],
            'label': 0.0 if x['id'] not in kaggle_imgs_labels else 1.0
            } for x in tqdm(kaggle_data['images'][0:ms])]

        kl = len(self.kaggle_imgs_list)
        al = len(self.aims_imgs_list)

        self.aims_imgs_list = list(filter(lambda x: osp.exists(x['file_name']), self.aims_imgs_list))
        self.kaggle_imgs_list = list(filter(lambda x: osp.exists(x['file_name']), self.kaggle_imgs_list))

        print("lost images", kl - len(self.kaggle_imgs_list), al - len(self.aims_imgs_list))

        self.shortest_list = min(len(self.aims_imgs_list), len(self.kaggle_imgs_list))

        print("number of images", len(self.aims_imgs_list), len(self.kaggle_imgs_list))

        # self.shortest_list = 10000 # TODO

        assert self.shortest_list > 0, "shortest dataset should have more than 0 existing images!"

    def __len__(self):
        # return len(self.kaggle_imgs_list)
        return self.shortest_list

    def __getitem__(self, index) -> tuple:
        # so that we don't always pair the same aims / kaggle image
        aims_index = np.random.randint(low=0, high=self.shortest_list)
        kaggle_index = np.random.randint(low=0, high=self.shortest_list)

        aims_img = Image.open(self.aims_imgs_list[aims_index]['file_name'])
        kaggle_img = Image.open(self.kaggle_imgs_list[kaggle_index]['file_name'])

        kaggle_label = self.kaggle_imgs_list[kaggle_index]['label']
        aims_label = self.aims_imgs_list[aims_index]['label']

        return self.transform_kaggle(kaggle_img), self.transform_aims(aims_img), torch.tensor([kaggle_label]), torch.tensor(aims_label)
