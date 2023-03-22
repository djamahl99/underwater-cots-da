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
from configs.train_config import DATA_DIRECTORY_SOURCE, DATA_DIRECTORY_TARGET, SOURCE_CATEGORY_ID, TARGET_CATEGORY_ID

def process_ann_box(ann: dict, image: dict, size: tuple) -> np.array:
    """Processes an annotation, resizing it to the given size from it's original size in image.

    Args:
        ann (dict): the annotation from the coco dictionary
        image (dict): the image from the coco dictionary
        size (tuple): the size of the image (height, width)

    Returns:
        np.array: processed bounding box
    """
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

    assert bbox[0::2].max() <= size[1] and bbox[1::2].max() <= size[0],\
         f"bbox out of bounds after rescale box={bbox} siz={size} old_size={old_size} ratios={ratios}"

    return bbox

class kaggle_aims_pair_boxed(data.Dataset):
    def __init__(self, aims_split="train.json", kaggle_split="mmdet_split_train.json", shortest=False, apply_augs=True) -> None:
        super().__init__()

        self.kaggle_split = kaggle_split
        self.aims_split = aims_split
        self.shortest = shortest

        p_default = 1.0 if apply_augs else 0.0

        self.size = (768, 1280)
        
        self.transform_kaggle = T.Compose(transforms=[
            T.Resize(self.size),
            T.RandomPosterize(4, p=min(0.5, p_default)),
            T.RandomAdjustSharpness(0.2, p=min(0.75, p_default)),
            T.RandomAutocontrast(p=min(0.6, p_default)),
            T.RandomEqualize(p=min(0.5, p_default)),
            T.ToTensor()
        ])

        self.transform_aims = T.Compose(transforms=[
            T.Resize(self.size),
            T.RandAugment(num_ops=2, magnitude=5),
            T.ToTensor()
        ])

        if not apply_augs:
            self.transform_aims = T.Compose(transforms=[
                T.Resize(self.size),
                T.ToTensor()
            ])

        self.kaggle_root = DATA_DIRECTORY_SOURCE
        self.aims_root = DATA_DIRECTORY_TARGET

        self.kaggle_anns = str(Path(self.kaggle_root) / f"annotations/{kaggle_split}")
        self.aims_anns = str(Path(self.aims_root) / f"annotations/{aims_split}")

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

        kl = len(self.kaggle_imgs_list)
        al = len(self.aims_imgs_list)

        self.aims_imgs_list = list(filter(lambda x: osp.exists(x['file_name']), self.aims_imgs_list))
        self.kaggle_imgs_list = list(filter(lambda x: osp.exists(x['file_name']), self.kaggle_imgs_list))

        print("lost images (aims, kaggle)", al - len(self.aims_imgs_list), kl - len(self.kaggle_imgs_list))

        self.shortest_list = min(len(self.aims_imgs_list), len(self.kaggle_imgs_list))

        print("number of images (aims, kaggle)", len(self.aims_imgs_list), len(self.kaggle_imgs_list))

        assert self.shortest_list > 0, "shortest dataset should have more than 0 existing images!"

        self.smallest = "aims"
        if len(self.aims_imgs_list) > len(self.kaggle_imgs_list):
            self.smallest = "kaggle"

    def __len__(self):
        return self.shortest_list
    
    def __getitem__(self, index) -> tuple:
        if self.smallest == "aims":
            aims_index = np.random.randint(low=0, high=len(self.aims_imgs_list))
            kaggle_index = index
        else:
            aims_index = index
            kaggle_index = np.random.randint(low=0, high=len(self.kaggle_imgs_list))

        aims_img = Image.open(self.aims_imgs_list[aims_index]['file_name'])
        kaggle_img = Image.open(self.kaggle_imgs_list[kaggle_index]['file_name'])

        kaggle_image_id = self.kaggle_imgs_list[kaggle_index]['image_id']
        aims_image_id = self.aims_imgs_list[aims_index]['image_id']

        return self.transform_kaggle(kaggle_img), self.transform_aims(aims_img), kaggle_image_id, aims_image_id
