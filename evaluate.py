import mmdet
import torch
from torch.utils import data
from tqdm import tqdm
from dataset.box_dataset import box_dataset

from network import WrappedYOLO, WrappedDetector
from network.batch_norm import set_bn_online 
import argparse
from torch import nn
import numpy as np

from configs.eval_config import get_arguments, splits, roots, DATA_DIRECTORY_SOURCE
from evaluation.box_dataset_evaluator import evaluate, evaluate_files

if __name__ == "__main__":
    args = get_arguments()

    assert args.dataset in ['aims_sep', 'aims_oct', 'kaggle'], "invalid dataset"
    
    split = splits[args.dataset][args.split]
    root = roots[args.dataset]

    ds = box_dataset(split=split, root=root)

    if args.subset > 0:
        # subset
        import numpy as np
        d = ds
        ds = data.Subset(ds, indices=list(np.random.randint(0, len(ds), size=args.subset)))
        ds.imgs_boxes = d.imgs_boxes
        ds.split = d.split
        ds.anns = d.anns
        ds.size = d.size

    if args.model == "yolov5":
        model = WrappedYOLO()
    elif args.model == "yolov8":
        model = WrappedDetector(config="yang_model\yolov8_l_Kaggle.py", ckpt="yang_model\yolov8_bbox_mAP_epoch_23.pth")
    elif args.model == "fasterrcnn":
        model = WrappedDetector(config="yang_model/faster-rcnn_r50_fpn_1x_cots.py", ckpt="yang_model/fasterrcnn_epoch_6.pth")
    else:
        raise Exception("bad model")

    if args.ckpt is not None:
        sd = torch.load(args.ckpt)

        if "backbone.stem.bn.running_mean" not in sd: # model was online :(
            sd['backbone.stem.bn.running_mean'] = torch.zeros((model.backbone.stem.bn.num_features))
            sd['backbone.stem.bn.running_var'] = torch.ones((model.backbone.stem.bn.num_features))

            if not args.online:
                print("model was set in online mode! Must be ran in online!")

        model.load_state_dict(sd)

    if args.online:
        set_bn_online(model)

    model = model.to(torch.device("cuda"))

    gt_filename = "gt.json"
    pred_filename = "pred.json"

    evaluate(model, ds, device=torch.device("cuda"), gt_filename=gt_filename, pred_filename=pred_filename)
    results_per_metric = evaluate_files(gt_filename=gt_filename, pred_filename=pred_filename)

    print(results_per_metric)