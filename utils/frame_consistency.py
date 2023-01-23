import torch

from torch.utils.data import DataLoader
from network.yolo_wrapper import WrappedYOLO
from network.relighting import LightNet
from dataset.kaggle_aims_pair_dataset_w_boxes import kaggle_aims_pair_boxed
from tqdm import tqdm

import matplotlib.pyplot as plt
import torchvision

def step(model, images, image_ids, gt_instances_prev, buffer, prev_id, enhancement=None):
    device = torch.device("cuda")
    current_id = image_ids[0].item()
    h, w = images.shape[-2:]

    if prev_id is not None and abs(current_id - prev_id) > 1:
        print("id diff large", prev_id, current_id)
        raise Exception("id large diff")

    loss = 0.0
    if prev_id is not None:
        img_metas = [dict(ori_shape=(h, w), scale_factor=1, batch_input_shape=(h,w)) for i in range(len(images))]

        with torch.no_grad():
            enhanced_images = images

            if enhancement is not None:
                # print("no enhancement")
                enhanced_images = images + enhancement(images)
                torchvision.utils.save_image(enhanced_images.detach(), "enhanced.jpg")

            losses = model(enhanced_images, instance_datas=gt_instances_prev.clone().to(device, dtype=torch.float32), img_metas=img_metas)

        loss = losses['loss_cls'] + losses['loss_obj'] + losses['loss_bbox']

        buffer.append(loss.item())

    boxes_current, scores_current = model.forward_pred_no_grad(images)
    boxes_current, scores_current = boxes_current[0], scores_current[0] # only one image
    num_boxes = boxes_current.shape[0]

    gt_instances = torch.zeros((num_boxes, 6), dtype=torch.float32)

    for j, box in enumerate(boxes_current):
        gt_instances[j, :] = torch.tensor([0.0, 1.0, *box], dtype=torch.float32)

    gt_instances_prev = gt_instances
    prev_id = current_id

    return gt_instances_prev, buffer, prev_id

def ema(buffer, v):
    buffer_ = [buffer[0]]
    value = buffer[0]
    for i in range(1, len(buffer)):
        value = value * v + (1 - v) * buffer[i]

        buffer_.append(value)

    assert len(buffer_) == len(buffer)
    
    del buffer
    return buffer_

def main():
    test_ds = kaggle_aims_pair_boxed(kaggle_split="mmdet_split_test.json", aims_split="instances_default.json")
    loader = DataLoader(test_ds, batch_size=1, shuffle=False)
    device = torch.device("cuda")

    model = WrappedYOLO(config="yang_model/yolov5_l_kaggle_cots_low_thresh.py")

    lightnet = LightNet()
    sd = torch.load("different-moon-91_light_latest.pth")
    lightnet.load_state_dict(sd)
    lightnet.to(device)

    lightnet.eval()

    print("len", len(test_ds))

    prev_id_kaggle = None
    prev_id_aims = None
    prev_id_bright_aims = None
    gt_instances_prev_kaggle = torch.tensor((0,6))
    gt_instances_prev_aims = torch.tensor((0,6))
    gt_instances_prev_bright_aims = torch.tensor((0,6))

    buffer_len = 1000
    buffer_kaggle = []
    buffer_aims = []
    buffer_bright_aims = []

    ema_p = 0.65

    for images_kaggle, images_aims, image_ids, _, _, aims_img_ids in tqdm(loader, desc="Evaluating"):
        images_kaggle = images_kaggle.to(device)
        images_aims = images_aims.to(device)

        gt_instances_prev_kaggle, buffer_kaggle, prev_id_kaggle = step(model, images_kaggle, image_ids, gt_instances_prev_kaggle, buffer_kaggle, prev_id_kaggle)
        gt_instances_prev_aims, buffer_aims, prev_id_aims = step(model, images_aims, aims_img_ids, gt_instances_prev_aims, buffer_aims, prev_id_aims)
        gt_instances_prev_bright_aims, buffer_bright_aims, prev_id_bright_aims = step(model, images_aims, aims_img_ids, gt_instances_prev_bright_aims, buffer_bright_aims, prev_id_bright_aims, enhancement=lightnet)


        if len(buffer_kaggle) >= buffer_len:
            # EMA
            buffer_kaggle = ema(buffer_kaggle, 0.8)
            buffer_aims = ema(buffer_aims, 0.8)
            buffer_bright_aims = ema(buffer_bright_aims, 0.8)

            plt.figure(figsize=(16,9))

            plt.plot(buffer_kaggle, ls='solid', label="Kaggle (dark)")
            plt.plot(buffer_aims, ls='dashed', label="Aims (dark)")
            plt.plot(buffer_bright_aims, ls='dotted', label="Stylized Aims -> Kaggle")

            plt.legend()

            plt.xlabel("Frame ($f$)")
            plt.ylabel("Loss")
            plt.title("Loss using frame $f - 1$ as ground truth, EMA smoothing=0.8")

            plt.show()
            exit()

if __name__ == "__main__":
    main()