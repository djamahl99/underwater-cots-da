from mmyolo.utils import register_all_modules
# from mmdet.utils import register_all_modules
from mmdet.apis import init_detector
import torch
from torch import nn
from mmengine.structures import InstanceData
from typing import List, Dict

class WrappedYOLO(nn.Module):
    def __init__(self, config=None, ckpt=None, device='cuda') -> None:
        super().__init__()

        register_all_modules()
        
        if config is None:
            config = "yang_model/yolov5_l_kaggle_cots.py"

        if ckpt is None:
            ckpt = "yang_model/bbox_mAP_epoch_200.pth"

        print("config, ckpt", config, ckpt)
        # ckpt = "yang_model/bbox_mAP_epoch_70.pth"

        model = init_detector(
                config=config,
                checkpoint=ckpt,
                device=device,
        )

        self.backbone = model.backbone
        self.neck = model.neck
        self.bbox_head = model.bbox_head

    def forward_pred_no_grad(self, x):
        h, w = x.shape[-2:]
        b = x.shape[0]
        with torch.no_grad():
            x = self.backbone(x)
            x = self.neck(x)
            x = self.bbox_head(x)

        img_metas = [dict(ori_shape=(h, w), scale_factor=1, batch_input_shape=(h,w)) for _ in range(b)]
        # preds = self.bbox_head.predict_by_feat(x[0], x[1], x[2], img_metas,\
                # rescale=False, with_nms=True)
        preds = self.bbox_head.predict_by_feat(*x, batch_img_metas=img_metas,\
                rescale=False, with_nms=True)

        # get bboxes etc
        bboxes = []
        scores = []
        for i in range(len(preds)):
            bboxes_ = []
            scores_ = []
            for j in range(len(preds[i]['bboxes'])):
                if preds[i]['bboxes'][j].numel() > 0:
                    bboxes_.append(preds[i]['bboxes'][j].detach().cpu().numpy())
                    scores_.append(preds[i]['scores'][j].item())
            # if len(bboxes_) > 0:
            bboxes.append(bboxes_)
            scores.append(scores_)
        bboxes = [torch.tensor(b) for b in bboxes]
        scores = [torch.tensor(s) for s in scores]

        return bboxes, scores

    # def forward(self, x):
        # x = self.backbone(x)
        # x = self.neck(x)
        # x = self.head(x)

    def forward(self, imgs, instance_datas: List[InstanceData], img_metas: List[Dict]):
        # loss_inputs = x + ([dict(bboxes=[avg_box], labels=[0])], [img_metas[0]])
        # print("x", x.shape, len(instance_datas), len(img_metas))
        x = self.backbone(imgs)
        x = self.neck(x)
        x = self.bbox_head(x)

        loss_inputs = x + (instance_datas, img_metas)
        losses = self.bbox_head.loss_by_feat(*loss_inputs)
        return losses