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

class box_dataset(data.Dataset):
    def __init__(self, root, split, transform=None, size=None) -> None:
        super().__init__()
        
        if size is None:
            self.size = (768, 1280)
        else:
            self.size = size
        
        # add more later TODO: use mmyolo LetterResize ... transforms
        
        if transform is not None:
            self.transform = T.Compose(transforms=[
                T.Resize(self.size),
                transform,
                T.ToTensor()
            ])
        else:
            self.transform = T.Compose(transforms=[
                T.Resize(self.size),
                T.ToTensor()
            ])

        self.root = root
        self.split = split

        # just use train :)
        self.anns = str(Path(self.root) / f"annotations/{self.split}")

        with open(self.anns, 'r') as f:
            data = json.load(f)

        image_id_to_image = {x['id']: x for x in data['images']}

        imgs_labels = {}
        imgs_boxes = {}

        cots_cat_id = None
        for x in data['categories']:
            if x['name'] == "COTS":
                cots_cat_id = x['id']
                break

        assert cots_cat_id is not None, data['categories']

        for ann in tqdm(data['annotations']):
            if ann['category_id'] == cots_cat_id:
                imgs_labels[ann['image_id']] = 1.0

                if not ann['image_id'] in imgs_boxes:
                    imgs_boxes[ann['image_id']] = {'bboxes': [], 'labels': []}

                bbox = process_ann_box(ann, image_id_to_image[ann['image_id']], self.size)

                imgs_boxes[ann['image_id']]['bboxes'].append(bbox)
                imgs_boxes[ann['image_id']]['labels'].append(1)

        self.imgs_list = [{
            'image_id': x['id'], 
            'file_name': self.root + "/" + x['file_name'],
            'label': 0.0 if x['id'] not in imgs_labels else 1.0
        } for x in tqdm(data['images'])]

        self.imgs_boxes = imgs_boxes

        num_with_boxes = 0
        num_without = 0

        for x in self.imgs_list:
            if x['image_id'] in imgs_boxes:
                num_with_boxes += 1
            else: 
                num_without += 1

        print(f"with anns={num_with_boxes} without anns ={num_without}")

        kl = len(self.imgs_list)

        self.imgs_list = list(filter(lambda x: osp.exists(x['file_name']), self.imgs_list))

        print("lost images =", kl - len(self.imgs_list))

        print("number of images ", len(self.imgs_list))

        assert len(self.imgs_list) > 0, "shortest dataset should have more than 0 existing images!"

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index) -> tuple:
        index = index

        img = Image.open(self.imgs_list[index]['file_name'])

        image_id = self.imgs_list[index]['image_id']

        return self.transform(img), image_id
