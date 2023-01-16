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

def process_ann_box(ann: dict, image: dict, size: tuple):
    # rescale the box to the current size
    old_size = (image['height'], image['width'])
    ratios = (size[0]/old_size[0], size[1]/old_size[1])

    # coco 2 xyxy
    bbox = np.array(ann['bbox'])
    [xy1, wh] = np.split(bbox, (2))
    xy2 = xy1 + wh
    bbox = np.concatenate((xy1, xy2), axis=-1)

    # apply ratio
    bbox[0::2] *= ratios[1]
    bbox[1::2] *= ratios[0]

    # clip box
    bbox[0::2] = np.clip(bbox[0::2], 0.0, float(size[1]))
    bbox[1::2] = np.clip(bbox[1::2], 0.0, float(size[0]))

    assert bbox[0::2].max() <= size[1] and bbox[1::2].max() <= size[0], f"bbox OOB after rescale? {bbox} {size} {old_size} {ratios}"

    return bbox

class kaggle_aims_pair_boxed(data.Dataset):
    def __init__(self, aims_split="train.json") -> None:
        super().__init__()

        assert aims_split in ["train.json", "val.json", "test.json"]
        self.aims_split = aims_split

        # size = (288, 512) # for testing on tactile
        self.size = (768, 1280)
        # self.size = (1088, 1920)
        
        # add more later TODO: use mmyolo LetterResize ... transforms
        self.transform_kaggle = T.Compose(transforms=[
            #T.CenterCrop(1080),
            T.Resize(self.size),
            # T.RandomPerspective(distortion_scale=0.1),
            # T.RandomAutocontrast(p=0.1),
            # T.RandomEqualize(),
            # T.RandomAdjustSharpness(),
            # T.RandomRotation(degrees=1),
            # T.AugMix(),
            T.ToTensor()
        ])

        self.transform_aims = T.Compose(transforms=[
            T.Resize(self.size),
            T.ToTensor()
        ])

        # self.kaggle_root = "/mnt/storage/djamahl/data/Kaggle_1080_google_v1"
        self.kaggle_root = "/home/etc004/code/YOLOX/data/Kaggle_1080_google_v1"

        # self.aims_root = "/home/etc004/code/YOLOX/AIMS_data_test"
        # self.aims_root = "/mnt/d61-visage-data/work/datasets/"
        self.aims_root = "AIMS_data_test"

        # just use train :)
        self.kaggle_anns = str(Path(self.kaggle_root) / "annotations/mmdet_split_train.json")
        self.aims_anns = str(Path(self.aims_root) / f"annotations/{aims_split}")
        # self.aims_anns = "aims_sep22.json"

        with open(self.kaggle_anns, 'r') as f:
            kaggle_data = json.load(f)

        with open(self.aims_anns, 'r') as f:
            aims_data = json.load(f)

        kaggle_image_id_to_image = {x['id']: x for x in kaggle_data['images']}
        aims_image_id_to_image = {x['id']: x for x in aims_data['images']}

        kaggle_imgs_labels = {}
        kaggle_imgs_boxes = {}
        aims_imgs_boxes = {}
        aims_imgs_labels = {}

        cots_cat_id_kaggle = 0
        cots_cat_id_aims = 1

        for ann in tqdm(aims_data['annotations']):
            if ann['category_id'] == cots_cat_id_aims:
                aims_imgs_labels[ann['image_id']] = 1.0

                if not ann['image_id'] in aims_imgs_boxes:
                    aims_imgs_boxes[ann['image_id']] = {'bboxes': [], 'labels': []}

                bbox = process_ann_box(ann, aims_image_id_to_image[ann['image_id']], self.size)

                aims_imgs_boxes[ann['image_id']]['bboxes'].append(bbox)
                aims_imgs_boxes[ann['image_id']]['labels'].append(1)

        for ann in tqdm(kaggle_data['annotations']):
            if ann['category_id'] == cots_cat_id_kaggle:
                kaggle_imgs_labels[ann['image_id']] = 1.0

                if not ann['image_id'] in kaggle_imgs_boxes:
                    kaggle_imgs_boxes[ann['image_id']] = {'bboxes': [], 'labels': []}

                bbox = process_ann_box(ann, kaggle_image_id_to_image[ann['image_id']], self.size)

                kaggle_imgs_boxes[ann['image_id']]['bboxes'].append(bbox)
                kaggle_imgs_boxes[ann['image_id']]['labels'].append(1)

        ms = -1

        self.aims_imgs_list = [{
            'image_id': x['id'], 
            'file_name': self.aims_root + "/" + x['file_name'],
            'label': 0.0 if x['id'] not in aims_imgs_labels else 1.0
        } for x in tqdm(aims_data['images'][0:ms])]

        self.kaggle_imgs_list = [{
            'image_id': x['id'], 
            'file_name': self.kaggle_root + "/" + x['file_name'],
            'label': 0.0 if x['id'] not in kaggle_imgs_labels else 1.0,
            'boxes': {'bboxes': [], 'labels': []} if x['id'] not in kaggle_imgs_boxes else kaggle_imgs_boxes[x['id']]
        } for x in tqdm(kaggle_data['images'][0:ms])]

        self.kaggle_imgs_boxes = kaggle_imgs_boxes
        self.aims_imgs_boxes = aims_imgs_boxes

        num_with_boxes = 0
        num_without = 0

        for x in self.kaggle_imgs_list:
            if len(x['boxes']['bboxes']) > 0:
                num_with_boxes += 1
            else: 
                num_without += 1

        print(f"kaggle with anns={num_with_boxes} without anns ={num_without}")

        kl = len(self.kaggle_imgs_list)
        al = len(self.aims_imgs_list)

        self.aims_imgs_list = list(filter(lambda x: osp.exists(x['file_name']), self.aims_imgs_list))
        self.kaggle_imgs_list = list(filter(lambda x: osp.exists(x['file_name']), self.kaggle_imgs_list))

        print("lost images", kl - len(self.kaggle_imgs_list), al - len(self.aims_imgs_list))

        self.shortest_list = min(len(self.aims_imgs_list), len(self.kaggle_imgs_list))

        print("number of images", len(self.aims_imgs_list), len(self.kaggle_imgs_list))

        assert self.shortest_list > 0, "shortest dataset should have more than 0 existing images!"

    def __len__(self):
        if self.aims_split == "train.json":
            # go through all kaggle (training yolo)
            return len(self.kaggle_imgs_list)

        # go though all aims split (evaluating yolo on aims val/test)
        return len(self.aims_imgs_list)

    def __getitem__(self, index) -> tuple:
        if self.aims_split == "train.json":
            # so that we don't always pair the same aims / kaggle image
            aims_index = np.random.randint(low=0, high=self.shortest_list)
            # kaggle_index = np.random.randint(low=0, high=self.shortest_list)
            kaggle_index = index
        else:
            aims_index = index
            kaggle_index = index

        aims_img = Image.open(self.aims_imgs_list[aims_index]['file_name'])
        kaggle_img = Image.open(self.kaggle_imgs_list[kaggle_index]['file_name'])

        kaggle_label = self.kaggle_imgs_list[kaggle_index]['label']
        image_id = self.kaggle_imgs_list[kaggle_index]['image_id']
        if self.aims_split != "train.json":
            image_id = self.aims_imgs_list[aims_index]['image_id']

        aims_label = self.aims_imgs_list[aims_index]['label']

        return self.transform_kaggle(kaggle_img), self.transform_aims(aims_img), image_id, torch.tensor(kaggle_label), torch.tensor(aims_label)
