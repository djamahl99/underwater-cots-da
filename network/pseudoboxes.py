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

class PseudoBoxer(nn.Module):
    """
        This module uses the teacher to produce gt_instances for calculating the loss on the student.

        It also stitches previous confident predictions into current images to add new bounding boxes.

    Args:
        nn (_type_): _description_
    """

    # queue of previous confident predictions
    queue = []

    def __init__(self, teacher: WrappedYOLO, darknet: ResnetGenerator, score_threshold=0.5, nms_threshold=0.4, max_to_add=5) -> None:
        super().__init__()

        self.teacher = teacher
        self.darknet = darknet
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_to_add = max_to_add

        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

        self.student_aug = transforms.Compose(transforms=[
            transforms.RandomPosterize(7, p=0.3),
            transforms.RandomAdjustSharpness(0.3, p=0.3),
            transforms.RandomAutocontrast(p=0.3),
            transforms.RandomEqualize(p=0.3)
        ])

    @torch.no_grad()
    def add_queue_objects_to_images(self, images: torch.Tensor, for_student=False):
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
            print("could not add anything - empty queue")
            return None, torch.zeros((0))

        print(f"can add! queue={len(self.queue)}")

        images_with_crops = []
        num_to_add_per_img = [np.random.randint(1, 5) for _ in enumerate(images)]
        total_to_add = sum(num_to_add_per_img)

        gt_instances = torch.zeros((total_to_add, 6), dtype=torch.float32)
        box_i = 0
        
        for i, image in enumerate(images):
            image_with_crop = image.detach().clone().cpu()
            h, w = image_with_crop.shape[-2:]

            # add crops to image
            for _ in range(num_to_add_per_img[i]):
                obj = np.random.choice(self.queue)
                w_o, h_o = obj['wh']

                r1, r2 = np.random.random(2)
                x1, y1 = r1 * (w - w_o*2), r2 * (h - h_o*2)

                x1, y1 = int(x1), int(y1)

                b = [min(x1, w - 1), min(y1, h - 1), min(x1 + w_o, w - 1), min(y1 + h_o, h - 1)]
                b = [max(x, 0) for x in b]

                gt_instances[box_i, :] = torch.tensor([float(i), 1.0, *b], dtype=torch.float32)
                box_i += 1

                assert max(b[0],b[2]) <= w and min(b) >= 0 and max(b[1],b[3]) <= h, f"{b} oob h={h} w={w} {(h_o, w_o)}"

                # image_with_crop[:, b[1]:b[3], b[0]:b[2]] = obj['image']

                print("obj image", obj['image'].shape)

                pil_obj = np.array(self.to_pil(obj['image']))
                pil_src = np.array(self.to_pil(image_with_crop))
                
                im_mask = np.full(pil_obj.shape, 255, dtype = np.uint8)

                center = (x1 + w_o//2, y1 + h_o//2)

                out = cv2.seamlessClone(pil_obj, pil_src, im_mask, center, cv2.NORMAL_CLONE)

                image_with_crop = self.to_pil(out)

                if for_student:
                    image_with_crop = self.student_aug(image_with_crop)

                image_with_crop = self.to_tensor(image_with_crop)

                if for_student:
                    torchvision.utils.save_image(image_with_crop, f"student_{i}.jpg")

                # torchvision.utils.save_image(image_with_crop, f"image_w_crop_{i}.jpg")
                # torchvision.utils.save_image(image.detach().clone().cpu(), f"image_{i}.jpg")
                # torchvision.utils.save_image(obj['image'], f"obj_{i}.jpg")

            images_with_crops.append(image_with_crop.unsqueeze(0))

        images_with_crops = torch.cat(images_with_crops, dim=0)

        gt_instances = torch.cat([x.unsqueeze(0) for x in gt_instances], dim=0)

        return images_with_crops.to(images.device), gt_instances

    @torch.no_grad()
    def forward(self, images_aims: torch.Tensor, brighter_images_aims: torch.Tensor, img_ids: torch.Tensor):
        h, w = images_aims.shape[-2:]

        # generate pseudo boxes from brightened / original images
        boxes_bright, scores_bright = self.teacher.forward_pred_no_grad(brighter_images_aims)
        boxes, scores = self.teacher.forward_pred_no_grad(images_aims)
        boxes = [torch.cat((boxes[i], boxes_bright[i]), dim=0) for i in range(len(boxes))]
        scores = [torch.cat((scores[i], scores_bright[i]), dim=0) for i in range(len(boxes))]

        # NMS to remove overlapping stylized / non-stylized teacher boxes
        nms_indices = [nms_if_non_empty(bs, scores[i], 0.4) for i, bs in enumerate(boxes)]
        boxes = [boxes[i][idx] for i, idx in enumerate(nms_indices)]
        scores = [scores[i][idx] for i, idx in enumerate(nms_indices)]

        num_neg = sum((s < self.score_threshold).sum().item() for s in scores)
        num_total = sum(x.numel() for x in scores)
        num_pos = num_total - num_neg

        student_images = None
        gt_instances_pseudo = torch.zeros((num_pos, 6), dtype=torch.float32)

        box_i = 0
        for i in range(len(images_aims)):
            bboxes = boxes[i]

            # see if can be added to the queue
            img_id = img_ids[i].item()
            add_to_queue = img_id not in [x['img_id'] for x in self.queue]

            for j, box in enumerate(bboxes):
                if scores[i][j] < self.score_threshold:
                    continue

                b = [int(x) for x in box]
                b[0::2] = [min(w - 1, max(x, 0)) for x in b[0::2]]
                b[1::2] = [min(h - 1, max(x, 0)) for x in b[1::2]]

                wh = (b[2] - b[0], b[3] - b[1])

                if wh[0] < 40 or wh[1] < 40:
                    continue

                assert box[0::2].max() <= w and box[1::2].max() <= h

                gt_instances_pseudo[box_i, :] = torch.tensor([float(i), 1.0, *box], dtype=torch.float32)
                box_i += 1

                if add_to_queue:
                    assert max(b[0], b[2]) <= w and max(b[1], b[3]) <= h and min(b) >= 0, f"box oob {b} -> img {images_aims.shape}"
                    im_copy = images_aims[i].detach().clone().cpu()

                    self.queue.append(dict(
                        img_id=img_id,
                        image=im_copy[:, b[1]:b[3], b[0]:b[2]],
                        wh=wh,
                        score=scores[i][j]
                    ))

        student_images, gt_instances_crops = self.add_queue_objects_to_images(images_aims, for_student=True)

        if student_images is None:
            student_images = images_aims

        if gt_instances_pseudo.numel() > 0:
            gt_instances = torch.cat((gt_instances_pseudo, gt_instances_crops), dim=0)
        else:
            gt_instances = gt_instances_crops

        print("gt instances / crops / pseudo", gt_instances.shape, gt_instances_crops.shape, gt_instances_pseudo.shape)

        return student_images, gt_instances
