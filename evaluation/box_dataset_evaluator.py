import mmdet
import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm
# from dataset.kaggle_aims_pair_dataset_w_boxes import kaggle_aims_pair_boxed
from dataset.box_dataset import box_dataset
from network.relighting import ResnetGenerator
# from mmdet.datasets import CocoDataset
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import json
from torchvision.ops import box_iou
import numpy as np
from typing import Tuple, List, Dict, Optional

def to_ann(ann_id: int, img_id: int, cat_id: int, box: List[float], area: float, score: float = None) -> Dict:
    d = dict(
        id=ann_id,
        image_id=img_id,
        category_id=cat_id,
        segmentation=[],
        area=int(area), # TODO
        bbox=[float(x) for x in box],
        iscrowd=0,
        # attributes={
        #     "occluded": False,
        #     "rotation": 0.0,
        #     "track_id": 1,
        #     "keyframe": True
        # }
    )

    if score is not None:
        d['score'] = score

    return d

def preprocess_results_dict(ann_path: str, size: Tuple) -> Dict:
    """Processes annotation file and resizes images to be size. Filters out non-COTS annotations.

    Args:
        ann_path (str): annotation file path
        size (Tuple): the size to resize the images to

    Returns:
        Dict: _description_
    """
    results_coco = {}
    cats, imgs = [], []

    with open(ann_path, "r") as f:
        data_dict = json.load(f)

    # filter out COTS categories
    cats = list(filter(lambda x: x['name'] == "COTS", data_dict['categories']))
    results_coco['categories'] = cats
    
    for img in data_dict['images']:
        img_ = img
        img_['width'] = size[1]
        img_['height'] = size[0]
        imgs.append(img_)

    results_coco['images'] = imgs

    return results_coco

def process_relighting(images: torch.tensor, enhancement: torch.nn.Module) -> torch.tensor:
    """Processes images using some relighting network(s)

    Args:
        images (torch.tensor): batch to process
        enhancement (torch.nn.Module): enhancement network

    Returns:
        torch.tensor: enhanced images
    """
    if enhancement is not None:
        if isinstance(enhancement, ResnetGenerator):
            images = torch.clamp(images + enhancement(images), 0, 1)
        elif isinstance(enhancement, list):
            outs = []
            for m in enhancement:
                outs.append(m(images) + images)
            images = torch.cat(outs, dim=1)
        else: # some other network?
            print("running enhancement")
            images = torch.clamp(enhancement(images), 0, 1)

    return images

def xyxy2xywha(box: torch.tensor) -> Tuple[torch.Tensor]:
    """Convert xyxy bbox to xywh and area

    Args:
        box (torch.tensor): xyxy bbox

    Returns:
        Tuple[torch.Tensor]: xywh bbox, area
    """
    xy1, xy2 = box.split((2,2), dim=-1)

    wh = (xy2 - xy1).abs()
    area = wh[0] * wh[1]
    box = torch.cat((xy1, wh), dim=-1)

    return box, area

def evaluate(model: nn.Module, ds: box_dataset, enhancement: nn.Module=None, device: torch.device=None, gt_filename: str=None,\
              pred_filename: str=None, return_preds: bool=False, thr: float=None):
    """Evaluate the COTS detection performance of model on ds

    Args:
        model (nn.Module): model to evaluate
        ds (box_dataset): dataset to evaluate on
        enhancement (nn.Module, optional): image enhancement network. Defaults to None.
        device (torch.device, optional): device to load models to. Defaults to None.
        gt_filename (str, optional): the gt annotations filename (COCO format). Defaults to None.
        return_preds (bool, optional): whether to return the predictions. Defaults to False.
        thr (float, optional): score threshold. Defaults to None.

    Returns:
        float: AP50
    """
    if gt_filename is None:
        gt_filename = "gt_coco_format_single.json"

    if pred_filename is None:
        pred_filename = "eval_results_coco_format_single.json" 

    model.eval()
    ann_path = ds.anns
    imgs_boxes = ds.imgs_boxes

    loader = data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    if device is None:
        device = torch.device("cuda")
    model = model.to(device)

    results_coco = preprocess_results_dict(ann_path, ds.size)
    img_ids = [x['id'] for x in results_coco['images']]
    cat_id = results_coco['categories'][0]

    gt_anns, pred_anns = [], []
    ann_id = 0
    ann_id_gt = 0
    returned_vals = []
    
    # now run preds
    for images, image_ids in tqdm(loader, desc="Evaluating"):
        image_ids = image_ids
        images = images.to(device)

        # run relighting if provided
        process_relighting(images, enhancement)
        bboxes, scores = model.forward_pred_no_grad(images)
        bboxes = torch.cat(bboxes, dim=0)
        scores = torch.cat(scores, dim=0)
        
        if thr is not None:
            bboxes = bboxes[scores >= thr]
            scores = scores[scores >= thr]

        if return_preds and num_boxes == 0:
            return None, None
            
        num_boxes = sum([0 if img_id.item() not in imgs_boxes else len(imgs_boxes[img_id.item()]['bboxes']) for img_id in image_ids])
        gt_instances = torch.zeros((num_boxes, 6), dtype=torch.float32)

        box_i = 0
        for i, img_id in enumerate(image_ids):
            if img_id.item() not in imgs_boxes:
                continue

            bboxes = imgs_boxes[img_id.item()]['bboxes']

            for box in bboxes:
                box, area = xyxy2xywha(torch.tensor(box))
                
                assert box[0::2].max() <= ds.size[1] and box[1::2].max() <= ds.size[0]
                gt_instances[box_i, :] = torch.tensor([float(i), 1.0, *box], dtype=torch.float32)
                box_i += 1

                gt_anns.append(to_ann(ann_id_gt, img_id=img_id.item(), cat_id=cat_id, box=box, area=area))
                ann_id_gt += 1
        
        returned_vals = [gt_instances, (bboxes, scores)]

        for box_i, box in enumerate(bboxes):
            if box.shape[0] == 0:
                continue

            box, area = xyxy2xywha(box)
            img_id = image_ids[0].item()
            assert img_id in img_ids, "img id does not exist"

            pred_anns.append(to_ann(ann_id=ann_id, img_id=img_id,\
                                        cat_id=cat_id, box=box.detach().cpu().numpy(), \
                                        area=area, score=scores[box_i].item()))
            ann_id += 1

    results_coco['annotations'] = pred_anns
    gt_coco = results_coco.copy()
    gt_coco['annotations'] = gt_anns

    with open(pred_filename, "w") as f:
        json.dump(results_coco, f)

    with open(gt_filename, "w") as f:
        json.dump(gt_coco, f)

    c_eval = COCO(pred_filename)
    c_gt = COCO(gt_filename)

    eval = COCOeval(cocoDt=c_eval, cocoGt=c_gt, iouType='bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()

    if return_preds:
        return eval.stats[1], returned_vals

    # return AP50 (can use evaluate_files to get all)
    return eval.stats[1]


def calculate_score(preds: List[torch.Tensor], gts: List[torch.Tensor], iou_th: float) -> float:
    """Calculates the F2 score at a iou threshold.

    Args:
        preds (List[torch.Tensor]): list of tensor bboxes for each image
        gts (List[torch.Tensor]): list of tensor GT bboxes for each image 
        iou_th (float): iou threshold for metric calculation

    Returns:
        float: F2 metric at iou_th
    """
    global idx_to_id
    num_tp = 0
    num_fp = 0
    num_fn = 0
    i = 0
    for p, GT in zip(preds, gts):
        i += 1

        if len(p) and len(GT):
            gt = GT.clone()
            gt[:, 2] = gt[:, 0] + gt[:, 2]
            gt[:, 3] = gt[:, 1] + gt[:, 3]
            pp = p.clone()
            pp[:, 2] = pp[:, 0] + pp[:, 2]
            pp[:, 3] = pp[:, 1] + pp[:, 3]
            iou_matrix = box_iou(pp, gt)
            tp = len(torch.where(iou_matrix.max(0)[0] >= iou_th)[0])
            fp = len(p) - tp
            fn = len(torch.where(iou_matrix.max(0)[0] < iou_th)[0])
            num_tp += tp
            num_fp += fp
            num_fn += fn
        elif len(p) == 0 and len(GT):
            num_fn += len(GT)
        elif len(p) and len(GT) == 0:
            num_fp += len(p)
    score = 5 * num_tp / (5 * num_tp + 4 * num_fn + num_fp)

    return score

def convert_coco(filename: str, thr: float=None) -> Tuple[List[torch.Tensor], set]:
    """Converts a COCO annotations file to a tuple, consisting of a list of bbox tensors and a set of image ids

    Args:
        filename (str): the filename path of the annotations
        thr (float, optional): the score threshold (for filtering predictions). Defaults to None.

    Returns:
        Tuple[List[torch.Tensor], set]: the filtered tensor annotations for each image and image ids set
    """
    global idx_to_id
    with open(filename, 'r') as f:
        ds = json.load(f)

    # dict version
    imgIdToAnns = {x['id']: [] for x in ds['images']}
    idx_to_id = {i: x['id'] for i, x in enumerate(ds['images'])}

    image_ids = set(x['id'] for x in ds['images'])
    image_ids_anns = set(x['image_id'] for x in ds['annotations'])

    for ann in ds['annotations']:
        if thr is not None and ann['score'] < thr:
            continue
        imgIdToAnns[ann['image_id']].append(ann)

    # tensor version
    tensorAnns = []
    for i, img in enumerate(ds['images']):
        img_id = img['id']
        anns = imgIdToAnns[img_id]
        n = len(anns)

        # (x1, y1, w, h)
        t = torch.zeros((n, 4))

        for j, ann in enumerate(anns):
            bbox = ann['bbox']
            t[j, :] = torch.tensor(bbox, dtype=torch.float32)

        tensorAnns.append(t)


    assert len(tensorAnns) == len(imgIdToAnns.keys())

    return tensorAnns, image_ids

def evaluate_files(gt_filename: str, pred_filename: str, thr=None):
    """Takes a gt_filename and pred_filename and runs evaluation at a give threshold.

    Args:
        gt_filename (str): _description_
        pred_filename (str): _description_
        thr (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    pred_filename_tmp = pred_filename.replace(".json", ".tmp.json")

    with open(pred_filename, 'r') as f:
        ds = json.load(f)

    # filter
    if thr is not None:
        ds['annotations'] = [x for x in ds['annotations'] if x['score'] >= thr]

    with open(pred_filename_tmp, 'w') as f:
        json.dump(ds, f)

    c_eval = COCO(pred_filename_tmp)
    c_gt = COCO(gt_filename)

    # metric imp ####################################################################
    gt_tensors, gt_ids = convert_coco(gt_filename)
    pred_tensors, pred_ids = convert_coco(pred_filename_tmp)

    iou_thrs = np.arange(0.3, 0.85, 0.05)

    scores = [calculate_score(pred_tensors, gt_tensors, iou_th) for iou_th in iou_thrs]
    f2 = np.mean(scores)



    eval = COCOeval(cocoDt=c_eval, cocoGt=c_gt, iouType='csiro')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    print("eval", eval.stats)

    eval_map = [
        ('AP 0.3:0.8', 0),
        ('AP 0.50', 1),
        ('AR 0.3:0.8', 8),
    ]

    results = {v[0]: eval.stats[v[1]] for v in eval_map}
    results['F2 0.3:0.8'] = f2

    return results

# added CSIRO .3:.8 metrics 
class Params:
    '''
    Params for coco evaluation api
    '''
    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setCsiroParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.3, 0.8, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [1, 10, 100]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 32 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'small', 'medium', 'large']
        self.useCats = 1

    def setKpParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value
        self.iouThrs = np.linspace(.5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.recThrs = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        self.maxDets = [20]
        self.areaRng = [[0 ** 2, 1e5 ** 2], [32 ** 2, 96 ** 2], [96 ** 2, 1e5 ** 2]]
        self.areaRngLbl = ['all', 'medium', 'large']
        self.useCats = 1
        self.kpt_oks_sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62,.62, 1.07, 1.07, .87, .87, .89, .89])/10.0

    def __init__(self, iouType='segm'):
        if iouType == 'segm' or iouType == 'bbox':
            self.setDetParams()
        elif iouType == 'csiro':
            self.setCsiroParams()
            iouType = 'bbox' # use bbox method now
        elif iouType == 'keypoints':
            self.setKpParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType
        # useSegm is deprecated
        self.useSegm = None
