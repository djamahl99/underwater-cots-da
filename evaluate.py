import mmdet
import torch
from torch.utils import data
from tqdm import tqdm
from dataset.kaggle_aims_pair_dataset_w_boxes import kaggle_aims_pair_boxed

# from mmdet.datasets import CocoDataset
from network.yolo_wrapper import WrappedYOLO
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import json
# c = CocoDataset()

# print("c", c)

# coco = COCOeval()
# print("coco", coco)


def evaluate(yolo: WrappedYOLO, ds: kaggle_aims_pair_boxed):
    aims_ann = ds.aims_anns

    print(aims_ann)

    loader = data.DataLoader(
        ds, batch_size=1, shuffle=True, num_workers=1, pin_memory=True
    )

    device = torch.device("cuda")
    yolo = yolo.to(device)

    anns = []
    cats = []
    imgs = []

    results_coco = {}

    with open(aims_ann, "r") as f:
        aims_d = json.load(f)

    cats = aims_d['categories'][0]
    results_coco['categories'] = [cats]
    
    for img in aims_d['images']:
        img_ = img
        img_['width'] = ds.size[1]
        img_['height'] = ds.size[0]
        imgs.append(img_)

    results_coco['images'] = imgs

    img_ids = [x['id'] for x in imgs]

    ann_id = 0

    import os
    if not os.path.exists("eval_results_coco_format.json"):

        # now run preds
        for _, images_aims, image_ids, _, _ in tqdm(loader, desc="Evaluating"):
            images_aims = images_aims.to(device)
            bboxes, scores = yolo.forward_pred_no_grad(images_aims)

            if bboxes.numel() == 0:
                continue
            
            bboxes = bboxes.flatten(0,-2)
            scores = scores.flatten(0,-1)

            for box_i, box in enumerate(bboxes):
                xy1, xy2 = box.split((2,2), dim=-1)

                wh = (xy2 - xy1).abs()
                area = wh[0] * wh[1]
                coco_bbox = torch.cat((xy1, wh), dim=-1)

                img_id = image_ids[0].item()

                assert img_id in img_ids, "img id does not exist"

                s = float(scores[box_i].item())

                if s < 0.5:
                    continue

                anns.append(dict(
                    id=ann_id,
                    image_id=image_ids[0].item(),
                    category_id=1,
                    segmentation=[],
                    area=int(area),
                    score=float(scores[box_i].item()),
                    # score=1.0,
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

        with open("eval_results_coco_format.json", "w") as f:
            json.dump(results_coco, f)

    # open again :)
    c_eval = COCO("eval_results_coco_format.json")
    # c_eval = COCO(aims_ann)
    c_gt = COCO(aims_ann)

    print("nun ans dt", len(c_eval.anns))
    print("nun ans gt", len(c_gt.anns))

    eval = COCOeval(cocoDt=c_eval, cocoGt=c_gt, iouType='bbox')
    print("evaluate()", eval.evaluate())
    print("accumulate", eval.accumulate())
    print("summarise", eval.summarize())

    
if __name__ == "__main__":
    ds = kaggle_aims_pair_boxed(aims_split="test.json")

    model = WrappedYOLO().eval()
    evaluate(model, ds)