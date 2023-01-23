import torch
from torch.utils.data import Subset
from evaluate import evaluate

from dataset.kaggle_aims_pair_dataset_w_boxes import kaggle_aims_pair_boxed
from network.yolo_wrapper import WrappedYOLO

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_agg import FigureCanvasAgg

from PIL import Image


import numpy as np

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
    
line_styles = ['--', '-.', '-']

def plot_preds(boxes, scores, ax, label_prefix="", col_idx=0):
    for i, bbox in enumerate(boxes):
        score = scores[i]
        bbox = xyxy2matplotlibxywh(bbox)

        label = f"{label_prefix} {score:.2f}"

        if col_idx == -1:
            col_idx = i

        print("pred", bbox)

        ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], alpha=1, ls=line_styles[col_idx % len(line_styles)], fill=False, lw=5, label=label, color=[c/255 for c in PALETTE[col_idx]]))

def plot_gt(gt_instances, ax, col_idx=2):
    for i, bbox in enumerate(gt_instances):
        bbox = bbox[2:]

        label = "GT"

        if col_idx == -1:
            col_idx = i

        print("gt", bbox)

        ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], alpha=1, ls=line_styles[col_idx % len(line_styles)], fill=False, lw=5, label=label, color=[c/255 for c in PALETTE[col_idx]]))        

def main():
#  evaluate(yolo: WrappedYOLO, ds: kaggle_aims_pair_boxed):

    model1 = WrappedYOLO()
    model2 = WrappedYOLO()
    sd = torch.load("different-moon-91_yolo_latest.pth")
    model2.load_state_dict(sd)


    # for param_q, param_k in zip(model1.backbone.parameters(), model2.backbone.parameters()):
    #     # param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
    #     print("max param diff", (param_q.data - param_k.data).abs().max())

    # exit()

    model1.eval()
    model2.eval()

    ds = kaggle_aims_pair_boxed(aims_split="val.json")

    min_val = 0.0001
    min_diff = 0.01

    print("len ds", len(ds))
    for idx in range(len(ds)):
        subset = Subset(ds, [idx])
        subset.aims_anns = ds.aims_anns
        subset.aims_imgs_boxes = ds.aims_imgs_boxes
        subset.kaggle_split = ds.kaggle_split
        subset.size = ds.size

        eval1, eval_preds1 = evaluate(model1, subset, True)
        eval2, eval_preds2 = evaluate(model2, subset, True)

        if eval1 < min_val and eval2 < min_val:
            continue

        if abs(eval1 - eval2) < min_diff:
            continue

        print("map 50 original", eval1)
        print("map50 now", eval2)

        gt_instances = eval_preds1[0]
        bboxes1, scores1 = eval_preds1[1]
        bboxes2, scores2 = eval_preds2[1]

        fig = plt.figure(figsize=(16,9))
        # canvas = FigureCanvasAgg(fig)

        # Do some plotting here
        ax = fig.add_subplot(111)

        _, images_aims, _, _, _, aims_img_ids = subset[0]
        print("images aims", images_aims.shape)

        ax.imshow(images_aims.permute(1, 2, 0).numpy())

        plot_preds(bboxes1, scores1, ax, label_prefix="orig.", col_idx=0)
        plot_preds(bboxes2, scores2, ax, label_prefix="UDA.", col_idx=1)
        plot_gt(gt_instances, ax)

        # canvas.draw()
        # buf = canvas.buffer_rgba()
        # # convert to a NumPy array
        # X = np.asarray(buf)

        # im = Image.fromarray(X).convert("RGB")
        # im.save(f"comparison_imgs/{idx}.jpg")
        plt.legend()

        fig.savefig(f"comparison_imgs/{aims_img_ids}.jpg")

        plt.close()

if __name__ == "__main__":
    main()