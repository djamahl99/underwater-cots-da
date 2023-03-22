import torch
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
import numpy as np
from typing import List

# line_styles = ['--', '-.', '-']
line_styles = ['-']

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
    """Converts xyxy bpx to xywh

    Args:
        x (tensor, np.array): bbox

    Returns:
        (tensor, np.array): xywh box for matplotlib Rectangle
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[0] = x[0]  # x left
    y[1] = x[1]  # y bottom
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height
    return y   


def plot_preds(boxes: torch.tensor, scores: torch.tensor, ax, label_prefix="", col_idx=0):
    for i, bbox in enumerate(boxes):
        score = scores[i]
        bbox = xyxy2matplotlibxywh(bbox)

        label = f"{label_prefix} {score:.2f}"

        if col_idx == -1:
            col_idx = i

        ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], alpha=1, ls=line_styles[col_idx % len(line_styles)], fill=False, lw=5, label=label, color=[c/255 for c in PALETTE[col_idx]]))

def plot_gt(gt_instances, ax, col_idx=2, label="GT", labels=None):
    set_col_idx = col_idx == -1
    for i, bbox in enumerate(gt_instances):
        bbox = bbox[2:]

        bbox = xyxy2matplotlibxywh(bbox)

        if labels is not None:
            label = labels[i]

        if set_col_idx:
            col_idx = i

        ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], alpha=1, ls=line_styles[col_idx % len(line_styles)], fill=False, lw=5, label=label, color=[c/255 for c in PALETTE[col_idx]]))        

def preds_to_dict(boxes, scores, gt_boxes):
    preds = [
        dict(
            boxes=boxes,
            scores=scores,
            labels=torch.zeros(boxes.shape[0])
        )
    ]

    target = [
        dict(
            boxes=gt_boxes,
            labels=torch.zeros(gt_boxes.shape[0])
        )
    ]

    return preds, target

def plot_student_pseudos(student_images: torch.tensor, gt_instances_student: List[torch.tensor], gt_instances_types, save_to=""):
    for img_id, img in enumerate(student_images):

        fig = plt.figure(figsize=(16,9))
        # canvas = FigureCanvasAgg(fig)

        # Do some plotting here
        ax = fig.add_subplot(111)
        print(gt_instances_types)

        ax.imshow(img.detach().cpu().permute(1, 2, 0).numpy())

        if gt_instances_student.shape[0] > 0:
            current_instances = gt_instances_student[gt_instances_student[:, 0] == img_id]
            current_types = []
            
            for i, gt in enumerate(gt_instances_student):
                if gt[0] == img_id:
                    current_types.append(gt_instances_types[i])

            plot_gt(current_instances, ax, col_idx=-1, labels=current_types)

        # canvas.draw()
        # buf = canvas.buffer_rgba()
        # # convert to a NumPy array
        # X = np.asarray(buf)

        # im = Image.fromarray(X).convert("RGB")
        # im.save(f"comparison_imgs/{idx}.jpg")
        plt.legend()
        plt.axis('off')
        fig.tight_layout()
        # plt.show()
        fig.savefig(f"{save_to}/student_boxes{img_id}.jpg", bbox_inches='tight')
        plt.close('all')