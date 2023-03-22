import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from dataset.kaggle_aims_pair_dataset_w_boxes import kaggle_aims_pair_boxed
from dataset.box_dataset import box_dataset
import numpy as np
from network import *
from pathlib import Path

from configs.train_config import get_arguments, yolo_loss_weights, yolov8_loss_weights, fasterrcnn_loss_weights, DATA_DIRECTORY_TARGET

import wandb

import torchvision
from network.online_batch_norm import BatchNormAdaptKDomain, set_bn_online, add_kdomain
from network.relighting import ResnetGenerator
from network.yolo_wrapper import WrappedYOLO
from network.mmdet_wrapper import WrappedDetector

from network.pseudoboxes import PseudoBoxer
from evaluation.box_dataset_evaluator import evaluate, evaluate_files

from visualisation.utils import plot_gt, plot_student_pseudos
import matplotlib.pyplot as plt

import PIL

def teacher_momentum(step_j: int):
    init_steps = 500
    num_warmup_steps = 1000
    lowest_momentum = 0.95
    highest_momentum = 0.999995
    # highest_momentum = 0.9995
    
    if step_j < init_steps:
        return lowest_momentum

    if step_j >= init_steps and step_j <= init_steps + num_warmup_steps:
        return ((step_j - init_steps)/num_warmup_steps) * (highest_momentum - lowest_momentum) + lowest_momentum
    
    return highest_momentum

def main():
    args = get_arguments()

    device = torch.device("cuda")

    cudnn.enabled = True
    cudnn.benchmark = True

    lightnet = LightNet(ngf=32) # aims -> kaggle
    darknet = LightNet() # kaggle -> aims

    saved_state_dict_l = torch.load(args.lightnet)
    saved_state_dict_d = torch.load(args.darknet)

    lightnet = nn.DataParallel(lightnet)
    lightnet.load_state_dict(saved_state_dict_l)
    lightnet = lightnet.module
    darknet.load_state_dict(saved_state_dict_d)

    lightnet.eval()
    lightnet.to(device)
    darknet.eval()
    darknet.to(device)

    for p in lightnet.parameters():
        p.requires_grad = False
    for p in darknet.parameters():
        p.requires_grad = False

    # yolo model
    if args.model == "yolov5":
        model = WrappedYOLO()
        teacher = WrappedYOLO()
        loss_weights = yolo_loss_weights
    elif args.model == "yolov8":
        model = WrappedDetector(config="yang_model\yolov8_l_Kaggle.py", ckpt="yang_model\yolov8_bbox_mAP_epoch_23.pth")
        teacher = WrappedDetector(config="yang_model\yolov8_l_Kaggle.py", ckpt="yang_model\yolov8_bbox_mAP_epoch_23.pth")
        loss_weights = yolov8_loss_weights
    elif args.model == "fasterrcnn":
        model = WrappedDetector(config="yang_model/faster-rcnn_r50_fpn_1x_cots.py", ckpt="yang_model/fasterrcnn_epoch_6.pth")
        teacher = WrappedDetector(config="yang_model/faster-rcnn_r50_fpn_1x_cots.py", ckpt="yang_model/fasterrcnn_epoch_6.pth")
        args.learning_rate_yolo = 1e-5
        loss_weights = fasterrcnn_loss_weights
    else:
        raise Exception("bad model")
    model.eval().to(device)

    # second teacher?
    teacherv8 = WrappedDetector(config="yang_model\yolov8_l_Kaggle.py", ckpt="yang_model\yolov8_bbox_mAP_epoch_23.pth")
    for p in teacherv8.parameters():
        p.requires_grad = False
    teacherv8.eval()

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    for p in model.parameters():
        p.requires_grad = False
    model.neck.eval()
    model.bbox_head.eval()

    for p in model.backbone.parameters():
        p.requires_grad = True
    model.backbone.train()

    # set_bn_online(model)
    # add_kdomain(teacher)
    set_bn_online(teacher)

    gt_instances_label_id = 0.0 if "8" in args.model else 1.0

    pseudoboxer = PseudoBoxer(teacher, teacherv8=teacherv8, max_queue=200, score_threshold=0.7, dropout=0.0, label_id=gt_instances_label_id)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    train_day_splits = ["day_split_20210907.json", "day_split_20210908.json", "day_split_20210909.json"]

    ds_val_aims = box_dataset(split="day_split_20210910_val.json", root=DATA_DIRECTORY_TARGET)

    args.learning_rate_yolo = 1e-6

    optimizer_model = optim.Adam([
        {'params': model.backbone.parameters(), 'lr': args.learning_rate_yolo}, 
        # {'params': model.neck.parameters(), 'lr': args.learning_rate_yolo}, 
        # {'params': model.bbox_head.parameters(), 'lr': args.learning_rate_yolo}, 
    ], lr=args.learning_rate_yolo, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    print("teacher momentums", [(x, teacher_momentum(x)) for x in np.linspace(0, 2000, 20)])

    pseudo_tradeoff = dict(
        pseudo=0.5,
        supervised=0.5
    )

    wandb.init(
        project="Underwater DA Multiday",
        name=args.run_name,
        config=dict(
            lr_yolo=args.learning_rate_yolo,
            lightnet_ckpt=args.lightnet,
            darknet_ckpt=args.darknet,
            loss_weights=loss_weights,
            model=args.model,
            score_thres=args.teacher_score_thresh
        )
    )

    if not os.path.exists(f"imgs/{wandb.run.name}/"):
        os.makedirs(f"imgs/{wandb.run.name}/")

    wandb.watch(models=[model.backbone])
    j = 0

    eval_every = 500

    for day_i, day_split in enumerate(train_day_splits):
        ds = kaggle_aims_pair_boxed(aims_split=day_split, shortest=False)
        trainloader = data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
        )
        # scheduler_model = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_model, T_0=100, T_mult=1, eta_min=args.learning_rate_yolo*0.01)
        scheduler_model = None

        day_j = 0
        with tqdm(desc=f"Day {day_i} {day_split}", unit='img') as pbar:
            for images_kaggle, images_aims, kaggle_image_ids, aims_img_ids in tqdm(trainloader, desc="Batch"):
                images_aims = images_aims.to(device)
                images_kaggle = images_kaggle.to(device)

                with torch.no_grad():
                    r = lightnet(images_aims)
                    brighter_images_aims = torch.clamp(images_aims + r, min=0.0, max=1.0) 

                    r = darknet(images_kaggle)
                    darker_images_kaggle = torch.clamp(images_kaggle + r, min=0.0, max=1.0) 

                student_images, gt_instances_pseudo, gt_instances_types, mean_psuedo_score = pseudoboxer(images_aims, brighter_images_aims, aims_img_ids)

                model.zero_grad()

                num_pseudo = sum(x == "pseudo" for x in gt_instances_types)
                num_crop = sum(x == "crop" for x in gt_instances_types)
                pbar.set_postfix(dict(num_pseudo=num_pseudo, num_crops=num_crop, mean_psuedo_score=mean_psuedo_score))
                pbar.update(images_aims.shape[0])

                h, w = images_kaggle.shape[-2:]
                img_metas = [dict(ori_shape=(h, w), scale_factor=1, batch_input_shape=(h,w)) for i in range(len(images_kaggle))]

                # (img_ind, labels, bbox_cx, bbox_cy, bbox_w, bbox_h)
                num_boxes = sum([0 if img_id.item() not in ds.kaggle_imgs_boxes else len(ds.kaggle_imgs_boxes[img_id.item()]['bboxes']) for img_id in kaggle_image_ids])
                gt_instances = torch.zeros((num_boxes, 6), dtype=torch.float32)

                box_i = 0
                for i, img_id in enumerate(kaggle_image_ids):
                    if img_id.item() in ds.kaggle_imgs_boxes:
                        bboxes = ds.kaggle_imgs_boxes[img_id.item()]['bboxes']

                        for box in bboxes:
                            assert box[0::2].max() <= ds.size[1] and box[1::2].max() <= ds.size[0]
                            gt_instances[box_i, :] = torch.tensor([float(i), gt_instances_label_id, *box], dtype=torch.float32)
                            box_i += 1

                losses_darker_kaggle = model(darker_images_kaggle, instance_datas=gt_instances.clone().to(device, dtype=torch.float32), img_metas=img_metas)

                for k in loss_weights:
                    losses_darker_kaggle[k] *= loss_weights[k]

                loss_yolo_kaggle_dark = sum(losses_darker_kaggle[k] for k in loss_weights)

                loss_yolo = loss_yolo_kaggle_dark

                ########################################################################################################
                # Pseudo-labeling training on aims (dark) dataset      #################################################
                ########################################################################################################
                losses_pseudolabeling = model(student_images, instance_datas=gt_instances_pseudo.clone().to(device, dtype=torch.float32), img_metas=img_metas)

                for k in loss_weights:
                    losses_pseudolabeling[k] *= loss_weights[k]

                loss_yolo_pseudo = sum(losses_pseudolabeling[k] for k in loss_weights)

                # backward with both supervised loss and unsupervised loss
                loss = pseudo_tradeoff['pseudo']*loss_yolo_pseudo + pseudo_tradeoff['supervised']*loss_yolo
                
                loss_yolo_total = loss.item()
                loss.backward()
                optimizer_model.step()

                teacher_diff = 0
                teacher_diff_n = 0

                # EMA update for the teacher
                with torch.no_grad():
                    m = teacher_momentum(day_j)
                    for param_q, param_k in zip(model.parameters(), teacher.parameters()):
                        teacher_diff += torch.mean(torch.abs(param_k.data - param_q.data))
                        teacher_diff_n += 1

                        param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

                if teacher_diff_n > 0:
                    teacher_diff /= teacher_diff_n


                if j % 50 == 0:
                    torchvision.utils.save_image(images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims.png")
                    torchvision.utils.save_image(images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle.png")
                    torchvision.utils.save_image(darker_images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_darker.png")
                    # torchvision.utils.save_image(aims_brightening.detach().cpu(), f"imgs/{wandb.run.name}/aims_brightening.png")
                    torchvision.utils.save_image(brighter_images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_brighter.png")
                    # torchvision.utils.save_image(kaggle_darkening.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_darkening.png")
                    torchvision.utils.save_image(student_images.detach().cpu(), f"imgs/{wandb.run.name}/student_images.png")

                    # try:
                    #     plot_student_pseudos(student_images, gt_instances_pseudo, gt_instances_types, save_to=f"imgs/{wandb.run.name}")
                    # except:
                    #     pass

                    if len(pseudoboxer.queue) > 0:
                        up_f = torch.nn.Upsample(size=(64,64))
                        queue_images = torch.cat([up_f(x['image'].unsqueeze(0)) for x in pseudoboxer.queue], dim=0)
                        torchvision.utils.save_image(queue_images, f"imgs/{wandb.run.name}/queue.png")


                pseudo_stats = pseudoboxer.get_score_stats()

                wandb.log(step=j, data={
                    'day': day_i,
                    'day_split': day_split,
                    'loss/yolo_pseudo': loss_yolo_pseudo.item(),
                    'loss/yoloboth': loss_yolo_total,
                    'loss/yolo_kaggle_dark': loss_yolo_kaggle_dark.item(),

                    'pseudoboxer/min_score': pseudo_stats[0],
                    'pseudoboxer/mean_score': pseudo_stats[1],
                    'pseudoboxer/max_score': pseudo_stats[2],
                    'pseudoboxer/queue_size': len(pseudoboxer.queue),
                    'pseudoboxer/score_threshold': pseudoboxer.score_threshold,

                    'teacher_momentum': teacher_momentum(day_j),
                    'teacher_diff': teacher_diff.item()
                })

                if scheduler_model is not None:
                    scheduler_model.step()
                    
                wandb.log(step=j, data={
                    'lr/yolo': optimizer_model.param_groups[0]['lr']
                })

                if (j % eval_every == 0 or j + 1 == len(trainloader)) and j > 0: # pre increment j
                    # Validation ###################################
                    pred_f_teacher = f"run_results/{wandb.run.name}/teacher_day_{day_i}_{j}.json"
                    pred_f_teacher8 = f"run_results/{wandb.run.name}/teacherv8_day_{day_i}_{j}.json"
                    pred_f_student = f"run_results/{wandb.run.name}/student_day_{day_i}_{j}.json"
                    pred_f_student_online = f"run_results/{wandb.run.name}/student_online_day_{day_i}_{j}.json"
                    gt_file = f"run_results/{wandb.run.name}/gt_day_{day_i}_{j}.json"

                    if not os.path.exists(Path(pred_f_teacher).parent):
                        os.makedirs(Path(pred_f_teacher).parent)

                    map50 = evaluate(model, ds_val_aims, pred_filename=pred_f_student, gt_filename=gt_file)
                    student_results = evaluate_files(pred_filename=pred_f_student, gt_filename=gt_file, thr=0.1)

                    model_online = WrappedYOLO()
                    model_online.load_state_dict(model.state_dict())
                    set_bn_online(model_online)

                    map50o = evaluate(model_online, ds_val_aims, pred_filename=pred_f_student_online, gt_filename=gt_file)
                    student_online_results = evaluate_files(pred_filename=pred_f_student_online, gt_filename=gt_file, thr=0.1)

                    map508 = evaluate(teacherv8, ds_val_aims, pred_filename=pred_f_teacher8, gt_filename=gt_file)
                    teacher8_results = evaluate_files(pred_filename=pred_f_teacher8, gt_filename=gt_file, thr=0.1)

                    del model_online

                    map50t = evaluate(teacher, ds_val_aims, pred_filename=pred_f_teacher, gt_filename=gt_file)
                    teacher_results = evaluate_files(pred_filename=pred_f_teacher, gt_filename=gt_file, thr=0.1)

                    wandb.log(step=j, data={
                        'val/AP50/student': map50,
                        'val/AP50/studentonline': map50o,
                        'val/AP50/teacher': map50t,
                        'val/AP50/teacher8': map508,

                        'val/F2 0.3:0.8/aims sdto': student_online_results['F2 0.3:0.8'],

                        'val/AP 0.3:0.8/aims sdto': student_online_results['AP 0.3:0.8'],
                        
                        'val/AR 0.3:0.8/aims sdto': student_online_results['AR 0.3:0.8'],

                        'val/F2 0.3:0.8/aims sdt': student_results['F2 0.3:0.8'],

                        'val/AP 0.3:0.8/aims sdt': student_results['AP 0.3:0.8'],

                        'val/AR 0.3:0.8/aims sdt': student_results['AR 0.3:0.8'],

                        'val/F2 0.3:0.8/aims tch': teacher_results['F2 0.3:0.8'],
                        'val/AP 0.3:0.8/aims tch': teacher_results['AP 0.3:0.8'],
                        'val/AR 0.3:0.8/aims tch': teacher_results['AR 0.3:0.8'],

                        'val/F2 0.3:0.8/aims tch': teacher8_results['F2 0.3:0.8'],
                        'val/AP 0.3:0.8/aims tch': teacher8_results['AP 0.3:0.8'],
                        'val/AR 0.3:0.8/aims tch': teacher8_results['AR 0.3:0.8'],

                        # 'val/kaggle/map50_teacher': map50k,
                    })
                    ################################################

                    # with torch.no_grad():
                    #     for param_q, param_k in zip(model.parameters(), teacher.parameters()):
                    #         param_q.data = param_k.detach().data.clone()

                    # print('taking snapshot ...')
                    # torch.save(model.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_yolo_{j}' + '.pth'))
                    # torch.save(teacher.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_teacher_{j}' + '.pth'))

                j += 1
                day_j += 1

        # if i_iter % args.save_pred_every == 0 and i_iter != 0:
        print('taking snapshot ...')
        torch.save(model.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_yolo_EOD_{day_i}.pth'))
        torch.save(teacher.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_teacher_EOD_{day_i}.pth'))

if __name__ == '__main__':
    main()
