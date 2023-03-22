from mmdet.apis import init_detector
# from mmdet.utils import register_all_modules
from mmyolo.utils import register_all_modules
import torch
from torch import nn
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from typing import List, Dict

def reduce(x):
    # print(f"reduce({x})")
    if isinstance(x, torch.Tensor):
        if x.numel() > 1:
            return torch.mean(x)
        return x
    elif isinstance(x, list):
        return torch.mean(torch.tensor(x))
    else:
        raise TypeError(f"x should be tensor or list but got {type(x)}")


class WrappedDetector(nn.Module):
    def __init__(self, config=None, ckpt=None, device='cuda') -> None:
        super().__init__()

        register_all_modules()

        if config is None:
            config = "yang_model/yolov5_l_kaggle_cots.py"

        if ckpt is None:
            ckpt = "yang_model/bbox_mAP_epoch_200.pth"

        print("config, ckpt", config, ckpt)

        model = init_detector(
                config=config,
                checkpoint=ckpt,
                device=device,
        ).train(True)

        self.model = model

    def forward_pred_no_grad(self, x):

        data_samples = []
        metainfo_dict = dict(ori_shape=(768, 1280), img_shape=(768, 1280), pad_shape=(768, 1280, 3), scale_factor=(1.0, 1.0))

        for i in range(len(x)):
            data_sample = DetDataSample(metainfo=metainfo_dict)
            data_samples.append(data_sample)

        preds = self.model.forward(inputs=x, data_samples=data_samples, mode='predict')
        
        bboxes = []
        scores = []
        for i in range(len(x)):
            bboxes.append(preds[i].pred_instances.bboxes.to(torch.device("cpu")))
            scores.append(preds[i].pred_instances.scores.to(torch.device("cpu")))

        return bboxes, scores

    def forward(self, imgs, instance_datas: List[InstanceData], img_metas: List[Dict]):
        gt_instances = instance_datas
        data_samples = []

        metainfo_dict = dict(ori_shape=(768, 1280, 3), img_shape=(768, 1280, 3), pad_shape=(768, 1280, 3), scale_factor=1.0)

        for i in range(len(imgs)):
            gt_instances_img = gt_instances[gt_instances[:, 0] == i]
            boxes = gt_instances_img[:, 2:].detach().clone()
            labels = torch.tensor(gt_instances_img[:, 1], dtype=torch.int64)

            data_sample = DetDataSample(metainfo=metainfo_dict)
            gt_insts = InstanceData(metainfo=metainfo_dict)
            gt_insts.bboxes = boxes
            gt_insts.labels = labels
            data_sample.gt_instances = gt_insts

            data_samples.append(data_sample)

        losses = self.model.forward(inputs=imgs, data_samples=data_samples, mode='loss')
        losses = {k: reduce(losses[k]) for k in losses.keys()}
        return losses
