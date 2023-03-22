import torch
import torch.nn as nn

from .yolo_wrapper import WrappedYOLO
from network.relighting import ResnetGenerator

from torchvision.ops import nms
import torchvision

from torchvision import transforms

import cv2
import numpy as np
from enum import IntEnum

class CropTypes(IntEnum):
    GT = 0
    NEG = 1 # probably don't use many of these..
    POS = 2

class PseudoFowardModes(IntEnum):
    ADVERSARIAL = 0
    YOLO = 1

def nms_if_non_empty(boxes, scores, nms_thresh):
    if len(boxes) == 0:
        # normal indices
        return slice(len(boxes))

    return nms(boxes, scores, nms_thresh)

def gkern(w, h, sig=1.):
    s = max(w, h)

    kern = gkernsquare(s, sig=sig)

    c = s // 2
    # return kern[(c - w//2):(c + w//2), (c - h//2):(c + h//2)]
    return kern[(c - h//2):(c + h//2), (c - w//2):(c + w//2)]

def gkernsquare(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    # ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    ax = np.linspace(-1.96, 1.96, l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)

    return kernel
    # print("kern", np.min(kernel), np.max(kernel))
    # return kernel / np.sum(kernel)

def random_box_size(obj: dict):
    # size_to_interval = {'small': (25, 32), 'medium': (32, 96), 'keep': None}
    # size_to_interval = {'small': (25, 32), 'medium': (32, 96), 'large': (96, 140), 'keep': None}
    size_to_interval = {'medium': (40, 96), 'large': (96, 120), 'keep': None}

    s_key = np.random.choice(list(size_to_interval.keys()))

    if s_key == 'keep':
        return obj

    side = np.random.random() * (size_to_interval[s_key][1] - size_to_interval[s_key][0]) + size_to_interval[s_key][0]

    w_o, h_o = obj['wh']
    s_o = min(w_o, h_o)
    r_o = side / s_o
    w_o_, h_o_ = int(r_o * w_o), int(r_o * h_o)

    assert min(w_o_, h_o_) >= 25, f"{(w_o_, h_o_)} too small"

    resize_f = torch.nn.Upsample(size=(h_o_, w_o_))
    out_image = resize_f(obj['image'].clone().unsqueeze(0))
    out_image = out_image.squeeze(0)

    out_obj = dict(
        img_id=obj['img_id'],
        image=out_image,
        wh=(w_o_, h_o_),
        score=obj['score']
    )
    
    return out_obj

class PseudoBoxer(nn.Module):
    """
        This module uses the teacher to produce gt_instances for calculating the loss on the student.

        It also stitches previous confident predictions into current images to add new bounding boxes.
    """

    # queue of previous confident predictions
    queue = []

    def __init__(self, teacher: WrappedYOLO, label_id=0.0, score_threshold=0.5, nms_threshold=0.5, max_queue=100, transparency=0.25, dropout=0.4, teacher2=None) -> None:
        super().__init__()

        self.teacher = teacher
        self.teacher2 = teacher2
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_queue = max_queue
        self.transparency = transparency
        self.dropout = dropout
        self.label_id = label_id

        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

        self.student_aug = transforms.Compose(transforms=[
            transforms.RandomPosterize(7, p=0.5),
            transforms.RandomAdjustSharpness(0.3, p=0.75),
            transforms.RandomAutocontrast(p=0.6),
            transforms.RandomEqualize(p=0.5)
        ])
        
    @torch.no_grad()
    def add_queue_objects_to_images(self, images: torch.Tensor, for_student=False, add_crops=True, neg_boxes=[]):
        """

        Generate images with copied objects to train generator / discriminator

        Args:
            images (torch.Tensor): images to add objects to 

        Returns:
            torch.Tensor: the images with added objects
        """
        labels = []
        if len(self.queue) == 0:
            # add no crops, return label for no crop
            # print("could not add anything - empty queue")
            return None, torch.zeros((0)), []

        # print(f"can add! queue={len(self.queue)}")

        images_with_crops = []
        num_to_add_per_img = [np.random.randint(1, 3) for _ in enumerate(images)]
        # num_to_add_per_img = [1 for _ in enumerate(images)]
        total_to_add = sum(num_to_add_per_img)

        if not add_crops:
            total_to_add = 0

        crop_scores = []
        gt_instances = torch.zeros((total_to_add, 6), dtype=torch.float32)
        box_i = 0
        
        for i, image in enumerate(images):
            image_with_crop = image.detach().clone().cpu()
            h, w = image_with_crop.shape[-2:]

            pil_src = np.array(self.to_pil(image_with_crop))
            out = None

            # add crops to image
            for crop_i in range(num_to_add_per_img[i]):
                if not add_crops or len(self.queue) == 0:
                    continue
                obj = np.random.choice(self.queue)

                # random resize object to medium/small/large COCO
                obj = random_box_size(obj)
                w_o, h_o = obj['wh']

                if len(neg_boxes[i]) > 0 and crop_i < len(neg_boxes[i]):
                    # use filtered out teacher preds
                    neg_box = neg_boxes[i][crop_i % len(neg_boxes[i])]
                    x1, y1 = [x.item() for x in neg_box[0:2]]

                    # check object is in frame
                    x1 = min(x1, w - w_o*2 - 10)
                    y1 = min(y1, h - h_o*2 - 10)
                    x1 = max(x1, w_o*2 + 10)
                    y1 = max(y1, h_o*2 + 10)
                else:
                    # random location
                    r1, r2 = np.random.random(2)
                    x1, y1 = r1 * (w - w_o*2), r2 * (h - h_o*2)

                x1, y1 = int(x1), int(y1)

                # check in bounds
                b = [min(x1, w - 1), min(y1, h - 1), min(x1 + w_o, w - 1), min(y1 + h_o, h - 1)]
                b = [max(x, 0) for x in b]
                assert max(b[0],b[2]) <= w and min(b) >= 0 and max(b[1],b[3]) <= h, f"{b} oob h={h} w={w} {(h_o, w_o)}"

                crop_scores.append(obj['score'])
                gt_instances[box_i, :] = torch.tensor([float(i), self.label_id, *b], dtype=torch.float32)
                box_i += 1

                pil_obj = np.array(self.to_pil(obj['image']))

                # generate mask
                mask = gkern(w=w_o, h=h_o, sig=2)
                im_mask = np.array(255*mask, dtype=np.uint8)
                
                # clone onto image
                center = (x1 + w_o//2, y1 + h_o//2)
                try:
                    out = cv2.seamlessClone(pil_obj, pil_src, im_mask, center, cv2.NORMAL_CLONE)
                except Exception as e:
                    print("clone gave an error, args:", pil_obj.shape, pil_src.shape, im_mask.shape, center)

                    print("e", e)
                pil_src = out

            if out is not None:
                image_with_crop = self.to_pil(out)
            else:
                image_with_crop = self.to_pil(image_with_crop)

            if for_student:
                image_with_crop = self.student_aug(image_with_crop)

            if not isinstance(image_with_crop, torch.Tensor):
                image_with_crop = self.to_tensor(image_with_crop)

            images_with_crops.append(image_with_crop.unsqueeze(0))

        images_with_crops = torch.cat(images_with_crops, dim=0)

        # injection smoother
        images_with_crops = images_with_crops.to(images.device)

        if total_to_add > 0:
            gt_instances = torch.cat([x.unsqueeze(0) for x in gt_instances], dim=0)

        return images_with_crops, gt_instances, crop_scores

    def get_score_stats(self):
        """
        Get min/mean/max stats for queue scores

        Returns:
            tuple: (min, mean, max)
        """
        scores = [x['score'] for x in self.queue]
        scores = torch.tensor(scores)

        if scores.numel() == 0:
            return 0.0, 0.0, 0.0

        return scores.min().item(), scores.mean().item(), scores.max().item()

    def prune_queue(self):
        """
        Prune the queue to remove less confident objects
        """
        scores = [x['score'] for x in self.queue]
        scores = torch.tensor(scores)

        # if len(self.queue) > self.max_queue and scores.min() > self.score_threshold:
        #     print(f"raising threshold to min={scores.min().item()}")
        #     self.score_threshold = scores.min().item()

        if len(self.queue) > self.max_queue and np.random.rand() <= self.dropout:
            # randomly remove from queue
            i_drop = np.random.choice(len(self.queue) - 1)
            print(f"PRUNING-DROPOUT: {self.queue[i_drop]['score']}")
            self.queue.pop(i_drop)

        if len(self.queue) > self.max_queue:
            median_score = torch.median(scores)
            print(f"PRUNING: median_score={median_score:.2f}, num_in_queue={len(self.queue)}")

            scores_topk = torch.topk(scores, k=self.max_queue).values
            print(f"min/max from topk {scores_topk.min()}/{scores_topk.max()}")

            topk_min = scores_topk.min()

            # now filter queue so that >= minimum in topk
            self.queue = list(filter(lambda x: x['score'] >= topk_min, self.queue))

    # @torch.no_grad()
    def forward(self, images_aims: torch.Tensor, brighter_images_aims: torch.Tensor, img_ids: torch.Tensor):
        h, w = images_aims.shape[-2:]

        # generate pseudo boxes from brightened / original images
        if self.teacher2 is not None:
            boxes_bright, scores_bright = self.teacher2.forward_pred_no_grad(images_aims)
            boxes, scores = self.teacher.forward_pred_no_grad(images_aims)
        else: # use relighting
            boxes_bright, scores_bright = self.teacher.forward_pred_no_grad(brighter_images_aims)
            boxes, scores = self.teacher.forward_pred_no_grad(brighter_images_aims)

        boxes = [torch.cat((boxes[i], boxes_bright[i]), dim=0) for i in range(len(boxes))]
        scores = [torch.cat((scores[i], scores_bright[i]), dim=0) for i in range(len(boxes))]

        # NMS to remove overlapping stylized / non-stylized teacher boxes
        nms_indices = [nms_if_non_empty(bs, scores[i], self.nms_threshold) for i, bs in enumerate(boxes)]
        boxes = [boxes[i][idx] for i, idx in enumerate(nms_indices)]
        scores = [scores[i][idx] for i, idx in enumerate(nms_indices)]

        # below threshold boxes and scores
        neg_boxes = [bs[scores[i] < self.score_threshold] for i, bs in enumerate(boxes)]
        neg_scores = [s[s < self.score_threshold] for s in scores]

        # don't want much overlap at all for injected crops
        nms_indices = [nms_if_non_empty(bs, neg_scores[i], 0.1) for i, bs in enumerate(neg_boxes)]
        neg_boxes = [neg_boxes[i][idx] for i, idx in enumerate(nms_indices)]
        neg_scores = [neg_scores[i][idx] for i, idx in enumerate(nms_indices)]

        num_neg_keep = 10
        neg_boxes = [x[:num_neg_keep] for x in neg_boxes]
        neg_scores = [x[:num_neg_keep] for x in neg_scores]

        # print("neg boxes 10", [list(zip(neg_boxes[i], neg_scores[i])) for i in range(len(neg_boxes))])

        num_neg = sum((s < self.score_threshold).sum().item() for s in scores)
        num_total = sum(x.numel() for x in scores)
        num_pos = num_total - num_neg

        student_images = None
        gt_instances_pseudo = torch.zeros((num_pos, 6), dtype=torch.float32)
        scores_pseudo = []

        box_i = 0
        for i in range(len(images_aims)):
            bboxes = boxes[i]

            # see if can be added to the queue
            img_id = img_ids[i].item()
            add_to_queue = img_id not in [x['img_id'] for x in self.queue]

            for j, box in enumerate(bboxes):
                if scores[i][j] < self.score_threshold: #or (scores[i][j] < self.score_threshold and np.random.rand() > scores[i][j]): # random chance accept low score
                    continue

                b = [int(x) for x in box]
                b[0::2] = [min(w - 1, max(x, 0)) for x in b[0::2]]
                b[1::2] = [min(h - 1, max(x, 0)) for x in b[1::2]]

                wh = (b[2] - b[0], b[3] - b[1])

                if wh[0] < 10 or wh[1] < 10:
                    num_pos -= 1
                    continue

                assert box[0::2].max() <= w and box[1::2].max() <= h

                scores_pseudo.append(scores[i][j].item())
                gt_instances_pseudo[box_i, :] = torch.tensor([float(i), self.label_id, *box], dtype=torch.float32)
                box_i += 1

                if add_to_queue:
                    assert max(b[0], b[2]) <= w and max(b[1], b[3]) <= h and min(b) >= 0, f"box oob {b} -> img {images_aims.shape}"
                    im_copy = images_aims[i].detach().clone().cpu()

                    self.queue.append(dict(
                        img_id=img_id,
                        image=im_copy[:, b[1]:b[3], b[0]:b[2]],
                        wh=wh,
                        score=scores[i][j].item()
                    ))
                
        self.prune_queue()

        student_images, gt_instances_crops, scores_crops = self.add_queue_objects_to_images(images_aims, for_student=True, neg_boxes=neg_boxes)

        scores_pseudo.extend(scores_crops)

        mean_score = 0.0 if len(scores_pseudo) == 0 else np.mean(scores_pseudo)


        if student_images is None:
            student_images = images_aims

        # in case got rid of some small boxes
        gt_instances_pseudo = gt_instances_pseudo[:num_pos]

        gt_instances_pseudo_non_zero = []

        for i in range(num_pos):
            if gt_instances_pseudo[i].abs().sum() > 0.001:
                gt_instances_pseudo_non_zero.append(gt_instances_pseudo[i].unsqueeze(0))

        if len(gt_instances_pseudo_non_zero) > 0:
            gt_instances_pseudo = torch.cat(gt_instances_pseudo_non_zero, dim=0)
        else:
            gt_instances_pseudo = torch.zeros((0, 6))

        if gt_instances_pseudo.numel() > 0: 
            gt_instances = torch.cat((gt_instances_pseudo, gt_instances_crops), dim=0)
        elif gt_instances_crops.numel() > 0:
            gt_instances = gt_instances_crops
        else:
            gt_instances = gt_instances_pseudo

        # print("gt instances / crops / pseudo / mean score", gt_instances.shape[0], gt_instances_crops.shape[0], gt_instances_pseudo.shape[0], mean_score)

        gt_instances_types = ["pseudo" for i in range(gt_instances_pseudo.shape[0])] + ["crop" for _ in range(gt_instances_crops.shape[0])]

        return student_images, gt_instances, gt_instances_types, mean_score

from torchvision.transforms.autoaugment import _apply_op
from torchvision.transforms import functional as F, InterpolationMode
import math
from enum import Enum
from typing import List, Tuple, Optional, Dict
from torch import Tensor


class RandAugment(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

        self.bbox_augs = ["Identity", "ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"]

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            # "Identity": (torch.tensor(0.0), False),
            # "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            # "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
            # "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
            # "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
            # "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor, gt_instances: Tensor) -> Tuple[Tensor]:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

            # print("op ", op_name)

            gt_instances = _apply_op_boxes(img, gt_instances, op_name, magnitude, interpolation=self.interpolation, fill=fill)
                

        return img, gt_instances

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s
    

def _apply_op_boxes(
    img, gt_instances: Tensor, op_name: str, magnitude: float, interpolation: InterpolationMode, fill: Optional[List[float]]
):
    if op_name == "ShearX":
        # magnitude should be arctan(magnitude)
        # official autoaug: (1, level, 0, 0, 1, 0)
        # https://github.com/tensorflow/models/blob/dd02069717128186b88afa8d857ce57d17957f03/research/autoaugment/augmentation_transforms.py#L290
        # compared to
        # torchvision:      (1, tan(level), 0, 0, 1, 0)
        # https://github.com/pytorch/vision/blob/0c2373d0bba3499e95776e7936e207d8a1676e65/torchvision/transforms/functional.py#L976
        gt_instances = affine_box(
            gt_instances,
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[math.degrees(math.atan(magnitude)), 0.0],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "ShearY":
        # magnitude should be arctan(magnitude)
        # See above
        gt_instances = affine_box(
            gt_instances,
            img,
            angle=0.0,
            translate=[0, 0],
            scale=1.0,
            shear=[0.0, math.degrees(math.atan(magnitude))],
            interpolation=interpolation,
            fill=fill,
            center=[0, 0],
        )
    elif op_name == "TranslateX":
        gt_instances = affine_box(
            gt_instances,
            img,
            angle=0.0,
            translate=[int(magnitude), 0],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "TranslateY":
        gt_instances = affine_box(
            gt_instances,
            img,
            angle=0.0,
            translate=[0, int(magnitude)],
            scale=1.0,
            interpolation=interpolation,
            shear=[0.0, 0.0],
            fill=fill,
        )
    elif op_name == "Rotate":
        raise Exception("not implemented")
        # img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)

    return gt_instances

import numbers

def concat_for_affine(tensor):
    return torch.cat((tensor, torch.ones((tensor.shape[0],1))), dim=-1).permute(dims=(1,0))

def affine_box(
    gt_instances: Tensor,
    img: Tensor,
    angle: float,
    translate: List[int],
    scale: float,
    shear: List[float],
    interpolation: InterpolationMode = InterpolationMode.NEAREST,
    fill: Optional[List[float]] = None,
    resample: Optional[int] = None,
    fillcolor: Optional[List[float]] = None,
    center: Optional[List[int]] = None,
) -> Tensor:
    
    gt_instances = gt_instances.clone()

    if isinstance(angle, int):
        angle = float(angle)

    if isinstance(translate, tuple):
        translate = list(translate)

    if isinstance(shear, numbers.Number):
        shear = [shear, 0.0]

    if isinstance(shear, tuple):
        shear = list(shear)

    if len(shear) == 1:
        shear = [shear[0], shear[0]]

    if len(shear) != 2:
        raise ValueError(f"Shear should be a sequence containing two values. Got {shear}")

    if center is not None and not isinstance(center, (list, tuple)):
        raise TypeError("Argument center should be a sequence")

    _, height, width = F.get_dimensions(img)
    if not isinstance(img, torch.Tensor):
        # center = (width * 0.5 + 0.5, height * 0.5 + 0.5)
        # it is visually better to estimate the center without 0.5 offset
        # otherwise image rotated by 90 degrees is shifted vs output image of torch.rot90 or F_t.affine
        if center is None:
            center = [width * 0.5, height * 0.5]
        # print("center", center)
        matrix = F._get_inverse_affine_matrix(center, angle, translate, scale, shear)
        # print("matrix pil", matrix, center, angle, translate, scale, shear)

        affine_matrix = torch.zeros((3,3))
        affine_matrix[0, 0:3] = torch.tensor(matrix[0:3], dtype=torch.float32)
        affine_matrix[1, 0:3] = torch.tensor(matrix[3:6], dtype=torch.float32)
        affine_matrix[2, 2] = 1.0

        # Rotation
        rotation_matrix = _get_rotation_matrix(angle)

        # # Scaling
        scaling_matrix = _get_scaling_matrix(scale)

        # Shear
        shear_matrix = _get_shear_matrix(shear[0], shear[1])

        # Translation
        translate_matrix = _get_translation_matrix(translate[0], translate[1])
        warp_matrix = (
            translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)
    
        warp_matrix = torch.tensor(warp_matrix)

        # print("affine matrix", affine_matrix, "warp", warp_matrix)

        # affine_matrix = warp_matrix

        gt_instances[:, 2:4] = (affine_matrix @ concat_for_affine(gt_instances[:, 2:4])).permute((1,0))[:, 0:2]
        gt_instances[:, 4:6] = (affine_matrix @ concat_for_affine(gt_instances[:, 4:6])).permute((1,0))[:, 0:2]
        
        # print("gt instances after", gt_instances)


        return gt_instances
        

    raise TypeError("not implmented for non-pil images" )
    center_f = [0.0, 0.0]
    if center is not None:
        _, height, width = F.get_dimensions(img)
        # Center values should be in pixel coordinates but translated such that (0, 0) corresponds to image center.
        center_f = [1.0 * (c - s * 0.5) for c, s in zip(center, [width, height])]

    translate_f = [1.0 * t for t in translate]
    print("center f translate f", center_f, translate_f)
    matrix = F._get_inverse_affine_matrix(center_f, angle, translate_f, scale, shear)
    print("matrix ", matrix)
    
    return gt_instances

def _get_rotation_matrix(rotate_degrees: float) -> np.ndarray:
    """Get rotation matrix.

    Args:
        rotate_degrees (float): Rotate degrees.

    Returns:
        np.ndarray: The rotation matrix.
    """
    radian = math.radians(rotate_degrees)
    rotation_matrix = np.array(
        [[np.cos(radian), -np.sin(radian), 0.],
            [np.sin(radian), np.cos(radian), 0.], [0., 0., 1.]],
        dtype=np.float32)
    return rotation_matrix

def _get_scaling_matrix(scale_ratio: float) -> np.ndarray:
    """Get scaling matrix.

    Args:
        scale_ratio (float): Scale ratio.

    Returns:
        np.ndarray: The scaling matrix.
    """
    scaling_matrix = np.array(
        [[scale_ratio, 0., 0.], [0., scale_ratio, 0.], [0., 0., 1.]],
        dtype=np.float32)
    return scaling_matrix

def _get_shear_matrix(x_shear_degrees: float,
                        y_shear_degrees: float) -> np.ndarray:
    """Get shear matrix.

    Args:
        x_shear_degrees (float): X shear degrees.
        y_shear_degrees (float): Y shear degrees.

    Returns:
        np.ndarray: The shear matrix.
    """
    x_radian = math.radians(x_shear_degrees)
    y_radian = math.radians(y_shear_degrees)
    shear_matrix = np.array([[1, np.tan(x_radian), 0.],
                                [np.tan(y_radian), 1, 0.], [0., 0., 1.]],
                            dtype=np.float32)
    return shear_matrix

def _get_translation_matrix(x: float, y: float) -> np.ndarray:
    """Get translation matrix.

    Args:
        x (float): X translation.
        y (float): Y translation.

    Returns:
        np.ndarray: The translation matrix.
    """
    translation_matrix = np.array([[1, 0., x], [0., 1, y], [0., 0., 1.]],
                                    dtype=np.float32)
    return translation_matrix
