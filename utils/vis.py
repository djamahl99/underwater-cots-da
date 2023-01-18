import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import torch

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

def plot_inference(gt: tuple, ds, img_id, image, save=""):
    boxes, scores = gt

    box_num = 0
    #### PLOTTING
    plt.figure(figsize=(16,9))
    ax = plt.subplot(1, 1, 1)

    ax.imshow(image.permute(1, 2, 0).detach().cpu().numpy())

    for i, bbox in enumerate(gt[0]):
        score = scores[i]
        bbox = xyxy2matplotlibxywh(bbox)

        label = f"{score}"

        #add rectangle to plot
        ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], alpha=1, fill=False, lw=5, label=label, color=[c/255 for c in PALETTE[box_num]]))
        box_num += 1

    gt_boxes = ds.aims_imgs_boxes[img_id]['bboxes']

    for bbox in gt_boxes:
        bbox = xyxy2matplotlibxywh(bbox)

        label = f"GT"

        #add rectangle to plot
        ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], alpha=1, fill=False, lw=5, label=label, color=[c/255 for c in PALETTE[box_num]]))
    
        box_num += 1
    plt.legend()
    if save == "":
        plt.show()
    else:
        plt.savefig(save)

    plt.close()