import mmdet
import torch
from torch.utils import data
from tqdm import tqdm
from dataset.kaggle_aims_pair_dataset_w_boxes import kaggle_aims_pair_boxed

from network.relighting import ResnetGenerator
from network.basic_enhance import NormalizeAdapt

# from mmdet.datasets import CocoDataset
from network.yolo_wrapper import WrappedYOLO
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import json
import argparse

def evaluate(yolo: WrappedYOLO, ds: kaggle_aims_pair_boxed, return_preds=False, enhance=None):
    aims_test = True
    ann_path = ds.aims_anns
    imgs_boxes = ds.aims_imgs_boxes

    if ds.kaggle_split == "mmdet_split_test.json" or "val" in ds.kaggle_split:
        aims_test = False
        ann_path = ds.kaggle_anns
        imgs_boxes = ds.kaggle_imgs_boxes

    loader = data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    device = torch.device("cuda")
    yolo = yolo.to(device)

    anns = []
    cats = []
    imgs = []

    results_coco = {}

    with open(ann_path, "r") as f:
        data_dict = json.load(f)

    # cats = data_dict['categories'][0]
    cats = list(filter(lambda x: x['name'] == "COTS", data_dict['categories']))
    print("cats", cats)
    cat_id = cats[0]['id']
    results_coco['categories'] = cats
    
    for img in data_dict['images']:
        img_ = img
        img_['width'] = ds.size[1]
        img_['height'] = ds.size[0]
        imgs.append(img_)

    results_coco['images'] = imgs

    gt_anns = []

    img_ids = [x['id'] for x in imgs]

    ann_id = 0
    ann_id_gt = 0

    returned_vals = []

    # now run preds
    for images_kaggle, images_aims, image_ids, _, _, _ in tqdm(loader, desc="Evaluating"):
        if aims_test:
            images = images_aims.to(device)
        else:
            images = images_kaggle.to(device)

        images_aims = images_aims.to(device)
        images_kaggle = images_kaggle.to(device)


        if enhance is not None:
            if isinstance(enhance, ResnetGenerator):
                images = images + enhance(images)
            elif isinstance(enhance, NormalizeAdapt):
                images = enhance(images_aims, images_kaggle)
            else:
                raise Exception("unkown enhanced")

        num_boxes = sum([0 if img_id.item() not in imgs_boxes else len(imgs_boxes[img_id.item()]['bboxes']) for img_id in image_ids])
        gt_instances = torch.zeros((num_boxes, 6), dtype=torch.float32)

        box_i = 0
        for i, img_id in enumerate(image_ids):
            if img_id.item() in imgs_boxes:
                bboxes = imgs_boxes[img_id.item()]['bboxes']

                for box in bboxes:
                    box = torch.tensor(box)
                    xy1, xy2 = box.split((2,2), dim=-1)

                    wh = (xy2 - xy1).abs()
                    area = wh[0] * wh[1]
                    box = torch.cat((xy1, wh), dim=-1)
                    
                    assert box[0::2].max() <= ds.size[1] and box[1::2].max() <= ds.size[0]
                    gt_instances[box_i, :] = torch.tensor([float(i), 1.0, *box], dtype=torch.float32)
                    box_i += 1

                    gt_anns.append(dict(
                        id=ann_id_gt,
                        image_id=img_id.item(),
                        category_id=cat_id,
                        segmentation=[],
                        area=int(area), # TODO
                        # score=1.0,
                        bbox=[float(x) for x in box],
                        iscrowd=0,
                        attributes={
                            "occluded": False,
                            "rotation": 0.0,
                            "track_id": 1,
                            "keyframe": True
                        }
                    ))
                    ann_id_gt += 1
        
        bboxes, scores = yolo.forward_pred_no_grad(images)

        bboxes = torch.cat(bboxes, dim=0)
        scores = torch.cat(scores, dim=0)

        returned_vals = [gt_instances, (bboxes, scores)]

        for box_i, box in enumerate(bboxes):
            if box.shape[0] == 0:
                continue
            xy1, xy2 = box.split((2,2), dim=-1)

            wh = (xy2 - xy1).abs()
            area = wh[0] * wh[1]
            coco_bbox = torch.cat((xy1, wh), dim=-1)

            img_id = image_ids[0].item()

            assert img_id in img_ids, "img id does not exist"

            anns.append(dict(
                id=ann_id,
                image_id=image_ids[0].item(),
                category_id=cat_id,
                segmentation=[],
                area=int(area),
                score=float(scores[box_i].item()),
                bbox=[float(x) for x in coco_bbox.detach().cpu().numpy()],
                iscrowd=0,
                attributes={
                    "occluded": False,
                    "rotation": 0.0,
                    "track_id": 1,
                    "keyframe": True
                }
            ))

            ann_id += 1


    results_coco['annotations'] = anns
    gt_coco = results_coco.copy()
    gt_coco['annotations'] = gt_anns

    # TODO: temp file name
    with open("eval_results_coco_format.json", "w") as f:
        json.dump(results_coco, f)

    with open("gt_coco_format.json", "w") as f:
        json.dump(gt_coco, f)

    # open again :)
    c_eval = COCO("eval_results_coco_format.json")
    c_gt = COCO("gt_coco_format.json")

    eval = COCOeval(cocoDt=c_eval, cocoGt=c_gt, iouType='bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    print("eval", eval.stats)

    if return_preds:
        return eval.stats[1], returned_vals

    # map50
    return eval.stats[1]

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'Evaluation',
                    description = 'Evaluates the model on the given dataset with the specified split.')

    parser.add_argument('--dataset', default='aims', choices=['aims', 'kaggle'])
    parser.add_argument('--split', default='test', choices=['test', 'val'])
    args = parser.parse_args()

    assert args.dataset in ['aims', 'kaggle'], "invalid dataset"
    
    splits = {
        'aims': {'test': 'test.json', 'val': 'val.json'},
        'kaggle': {'test': 'mmdet_split_test.json', 'val': 'mmdet_split_val.json'}
    }

    split = splits[args.dataset][args.split]

    if args.dataset == "aims":
        ds = kaggle_aims_pair_boxed(aims_split=split)
    else:
        ds = kaggle_aims_pair_boxed(kaggle_split=split)

    model = WrappedYOLO().eval()
    map50 = evaluate(model, ds)