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

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# c = CocoDataset()

# print("c", c)

# coco = COCOeval()
# print("coco", coco)
PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
            (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
            (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
            (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
            (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
            (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
            (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
            (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
            (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
            (134, 134, 103), (145, 148, 174), (255, 208, 186),
            (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
            (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
            (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
            (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
            (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
            (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
            (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
            (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
            (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
            (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
            (191, 162, 208)]

def xyxy2matplotlibxywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = x[0]  # x left
    y[1] = x[1]  # y bottom
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height
    return y   


def evaluate(yolo: WrappedYOLO, ds: kaggle_aims_pair_boxed):
    aims_ann = ds.aims_anns

    print(aims_ann)

    loader = data.DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
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

    gt_anns = []

    img_ids = [x['id'] for x in imgs]

    ann_id = 0
    ann_id_gt = 0

    #### PLOTTING
    fig = plt.figure(figsize=(20,20))
    #define Matplotlib figure and axis
    # fig, ax = plt.subplots()

    import os
    if not os.path.exists("eval_results_coco_format.json") or True:

        # now run preds
        for _, images_aims, image_ids, _, _ in tqdm(loader, desc="Evaluating"):
            images_aims = images_aims.to(device)

            plt.clf()

            ax = plt.subplot(1,1,1)

            # ax.set_axis_off()
            # fig.add_axes(ax)

            ax.imshow(images_aims[0].permute(1, 2, 0).detach().cpu().numpy())
            ###############

            num_boxes = sum([0 if img_id.item() not in ds.aims_imgs_boxes else len(ds.aims_imgs_boxes[img_id.item()]['bboxes']) for img_id in image_ids])
            gt_instances = torch.zeros((num_boxes, 6), dtype=torch.float32)

            box_i = 0
            for i, img_id in enumerate(image_ids):
                if img_id.item() in ds.aims_imgs_boxes:
                    bboxes = ds.aims_imgs_boxes[img_id.item()]['bboxes']

                    for box in bboxes:
                        box = torch.tensor(box)
                        xy1, xy2 = box.split((2,2), dim=-1)

                        wh = (xy2 - xy1).abs()
                        area = wh[0] * wh[1]
                        box = torch.cat((xy1, wh), dim=-1)
                        
                        assert box[0::2].max() <= ds.size[1] and box[1::2].max() <= ds.size[0]
                        gt_instances[box_i, :] = torch.tensor([float(i), 1.0, *box], dtype=torch.float32)
                        box_i += 1

                        label = f"GT box {box_i}"
                        ax.add_patch(Rectangle((box[0], box[1]), box[2], box[3], alpha=1, fill=False, lw=1, label=label, color=[c/255 for c in PALETTE[0]]))


                        gt_anns.append(dict(
                            id=ann_id_gt,
                            image_id=img_id.item(),
                            category_id=1,
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
            
            bboxes, scores = yolo.forward_pred_no_grad(images_aims)
            
            bboxes = bboxes.flatten(0,-2)
            scores = scores.flatten(0,-1)

            for box_i, box in enumerate(bboxes):
                if box.shape[0] == 0:
                    continue
                xy1, xy2 = box.split((2,2), dim=-1)

                wh = (xy2 - xy1).abs()
                area = wh[0] * wh[1]
                coco_bbox = torch.cat((xy1, wh), dim=-1)

                img_id = image_ids[0].item()

                assert img_id in img_ids, "img id does not exist"

                label = f"pred {box_i}"

                ax.add_patch(Rectangle((coco_bbox[0].item(), coco_bbox[1].item()), coco_bbox[2].item(), coco_bbox[3].item(), alpha=1, fill=False, lw=1, label=label, color=[c/255 for c in PALETTE[1]]))

                anns.append(dict(
                    id=ann_id,
                    image_id=image_ids[0].item(),
                    category_id=1,
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

        with open("eval_results_coco_format.json", "w") as f:
            json.dump(results_coco, f)

        with open("gt_coco_format.json", "w") as f:
            json.dump(gt_coco, f)

    # open again :)
    c_eval = COCO("eval_results_coco_format.json")
    # c_eval = COCO(aims_ann)
    c_gt = COCO("gt_coco_format.json")

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