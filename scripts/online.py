from network.yolo_wrapper import WrappedYOLO
from dataset.kaggle_aims_pair_dataset_w_boxes import kaggle_aims_pair_boxed
from evaluate import evaluate

from torch import nn
import torch

from torch.utils.data import Subset

def main():
    yolo = WrappedYOLO()
    yolo.load_state_dict(torch.load("dual-discrim-epoch200-yang_yolo_4000.pth"))
    # torch.save(yolo, "yolo.pth")
    # ds_val_aims = kaggle_aims_pair_boxed(aims_split="val.json")
    ds_val_aims = kaggle_aims_pair_boxed(kaggle_split="mmdet_split_val.json")
    d = ds_val_aims
    # ds_val_aims = Subset(ds_val_aims, indices=[i for i in range(500)])
    import numpy as np
    ds_val_aims = Subset(ds_val_aims, indices=list(np.random.randint(0, len(d), size=500)))
    ds_val_aims.aims_anns = d.aims_anns
    ds_val_aims.kaggle_imgs_boxes = d.kaggle_imgs_boxes
    ds_val_aims.aims_imgs_boxes = d.aims_imgs_boxes
    ds_val_aims.kaggle_split = d.kaggle_split
    ds_val_aims.aims_split = d.aims_split
    ds_val_aims.kaggle_anns = d.kaggle_anns
    ds_val_aims.size = d.size
    # d.kaggle_imgs_boxes

    yolo.eval()

    print("before")
    ap50_1 = evaluate(yolo, ds_val_aims)

    # now evaluate with adaptive batch norm
    for m in yolo.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean = None
            m.running_var = None

    yolo = yolo.to(torch.device("cuda"))

    ap50_2 = evaluate(yolo, ds_val_aims)

    # assert that running_mean and var are none
    for m in yolo.modules():
        if isinstance(m, nn.BatchNorm2d):
            assert m.running_mean is None and m.running_var is None

    print(ap50_2)

if __name__ == "__main__":
    main()