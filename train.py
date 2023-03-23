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
import wandb
import torchvision

from configs.train_config import get_arguments, yolo_loss_weights, yolov8_loss_weights, fasterrcnn_loss_weights,\
                 D_DS, D_ADV, e_wgts, DATA_DIRECTORY_TARGET

from network import WrappedDetector, WrappedYOLO, BatchNormAdaptKDomain
from network.online_batch_norm import add_kdomain, set_bn_online
from network.relighting import L_ColorInvarianceConv, Loss_bounds, ResnetGenerator
from network.pseudoboxes import PseudoBoxer
from visualisation.utils import plot_gt, plot_preds, plot_student_pseudos
from evaluation.box_dataset_evaluator import evaluate, evaluate_files

def teacher_momentum(step_j: int):
    init_steps = 500
    num_warmup_steps = 1000
    lowest_momentum = 0.6
    # highest_momentum = 0.99995
    highest_momentum = 0.99995
    
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

    lightnet = LightNet(ngf=32)
    darknet = LightNet()
    
    saved_state_dict_l = torch.load(args.lightnet)
    saved_state_dict_d = torch.load(args.darknet)

    light_load = os.path.basename(args.lightnet)
    dark_load = os.path.basename(args.darknet)

    lightnet = nn.DataParallel(lightnet)
    lightnet.load_state_dict(saved_state_dict_l)
    lightnet = lightnet.module
    darknet.load_state_dict(saved_state_dict_d)

    lightnet.eval()
    lightnet.to(device)
    
    darknet.eval()
    darknet.to(device)

    if args.train_style:
        model_D_ds = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3) # classes = num input channels    
        saved_state_dict = torch.load(D_DS)
        model_D_ds.load_state_dict(saved_state_dict)

        model_D_ds.eval()
        model_D_ds.to(device)
    else:
        model_D_ds = None
  
    if args.train_style:
        model_D_adv = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3) # classes = num input channels    
        saved_state_dict = torch.load(D_ADV)
        model_D_adv.load_state_dict(saved_state_dict)
        model_D_adv.eval()
        model_D_adv.to(device)
    else:
        model_D_adv = None

    # yolo model
    if args.model == "yolov5":
        model = WrappedYOLO()
        teacher = WrappedYOLO()
        loss_weights = yolo_loss_weights
    elif args.model == "yolov8":
        model = WrappedDetector(config="yang_model/yolov8_l_Kaggle.py", ckpt="yang_model/yolov8_bbox_mAP_epoch_23.pth")
        teacher = WrappedDetector(config="yang_model/yolov8_l_Kaggle.py", ckpt="yang_model/yolov8_bbox_mAP_epoch_23.pth")
        loss_weights = yolov8_loss_weights
    elif args.model == "fasterrcnn":
        model = WrappedDetector(config="yang_model/faster-rcnn_r50_fpn_1x_cots.py", ckpt="yang_model/fasterrcnn_epoch_6.pth")
        teacher = WrappedDetector(config="yang_model/faster-rcnn_r50_fpn_1x_cots.py", ckpt="yang_model/fasterrcnn_epoch_6.pth")
        args.learning_rate_yolo = 1e-5
        loss_weights = fasterrcnn_loss_weights
    else:
        raise Exception("bad model")
    model.eval().to(device)

    # clip_value = 0.5
    # for p in model.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # set_bn_online(model)
    set_bn_online(teacher)
    # add_kdomain(teacher)

    gt_instances_label_id = 0.0 if "8" in args.model else 1.0

    pseudoboxer = PseudoBoxer(teacher, max_queue=75, score_threshold=args.teacher_score_thresh, dropout=0.0, label_id=gt_instances_label_id)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    ds = kaggle_aims_pair_boxed(aims_split="aims_sep22_10percent_split_train.json")
    trainloader = data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )

    ds_val_aims = box_dataset(split="aims_sep22_10percent_split_val.json", root=DATA_DIRECTORY_TARGET)

    optimizer_model = optim.Adam([
        {'params': model.parameters(), 'lr': args.learning_rate_yolo}, 
    ], lr=args.learning_rate_yolo, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    if args.train_style:
        optimizer_D = optim.Adam(list(model_D_ds.parameters()) + list(model_D_adv.parameters()), lr=args.learning_rate_D, betas=(0.9, 0.99))
        optimizer_dark = optim.Adam(darknet.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        optimizer_light = optim.Adam(lightnet.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    print("teacher momentums", [(x, teacher_momentum(x)) for x in np.linspace(0, len(ds), 20)])
    scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_model, T_max=len(trainloader), eta_min=0, last_epoch=-1, verbose=False)

    if args.train_style:
        scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda epoch: 0.8**(epoch))
        scheduler_dark = optim.lr_scheduler.LambdaLR(optimizer_dark, lr_lambda=lambda epoch: 0.8**(epoch))
        scheduler_light = optim.lr_scheduler.LambdaLR(optimizer_light, lr_lambda=lambda epoch: 0.8**(epoch))

    loss_exp_z = L_exp_z(32)
    loss_TV = L_TV()
    loss_SSIM = SSIM()
    loss_bce = nn.BCEWithLogitsLoss()
    loss_l2 = nn.MSELoss()
    gan_criterion = loss_l2
    loss_ci = L_ColorInvarianceConv(invariant='W')

    kaggle_label = 0.0
    aims_label = 1.0

    real_label = 1.0
    fake_label = 0.0

    no_crop_label = 0.0
    crop_label = 1.0

    pseudo_tradeoff = dict(
        pseudo=0.9,
        supervised=0.1
    )

    wandb.init(
        project="DANNet YOLO DA",
        name=args.run_name,
        config=dict(
            lr_yolo=args.learning_rate_yolo,
            loss_weights=loss_weights,
            model=args.model,
            score_thres=args.teacher_score_thresh,
            enhancement_weights=e_wgts,
            light_load=light_load,
            dark_load=dark_load
        )
    )

    if not os.path.exists(f"imgs/{wandb.run.name}/"):
        os.makedirs(f"imgs/{wandb.run.name}/")

    wandb.watch(models=[model.backbone])
    j = 0

    mean_light_ema = torch.tensor(0.5).to(device)
    ema_v = 0.99

    mean_dark_ema = torch.tensor(0.4).to(device)

    mean_lightnet_out = torch.tensor(0.5).to(device)
    mean_darknet_out = torch.tensor(0.5).to(device)

    for i_iter in range(args.num_steps):
        for images_kaggle, images_aims, kaggle_image_ids, aims_img_ids in tqdm(trainloader, desc="Batch"):
            images_aims = images_aims.to(device)
            images_kaggle = images_kaggle.to(device)
            with torch.no_grad():
                mean_light = images_kaggle.mean()
                mean_light_ema = mean_light_ema * ema_v + (1 - ema_v) * mean_light

                mean_dark = images_aims.mean()
                mean_dark_ema = mean_dark_ema * ema_v + (1 - ema_v) * mean_dark
            
            # to loss
            mean_light = mean_light_ema
            mean_dark = mean_dark_ema
            
            b_size = images_kaggle.shape[0]

            # decrease supervised weight
            if j >= 1000 and j % 500 == 0:
                pseudo_tradeoff['supervised'] *= 0.8
                pseudo_tradeoff['pseudo'] = 1.0 - pseudo_tradeoff['supervised'] 

            wandb.log(step=j, data={'tradeoff/supervised': pseudo_tradeoff['supervised'], 'tradeoff/pseudo': pseudo_tradeoff['pseudo']})

            if args.train_style: # TODO: separate this code
                ########################################################################################################
                # DISCRIMINATOR 
                ########################################################################################################
                for p in lightnet.parameters():
                    p.requires_grad_(False)
                for p in darknet.parameters():
                    p.requires_grad_(False)
                for p in model.parameters():
                    p.requires_grad_(False)
                for p in model_D_ds.parameters():
                    p.requires_grad_(True)
                for p in model_D_adv.parameters():
                    p.requires_grad_(True)
                
                model_D_ds.zero_grad()
                model_D_adv.zero_grad()
                #model.zero_grad()

                # aims -> kaggle
                r = lightnet(images_aims)
                if isinstance(lightnet, ResnetGenerator):
                    brighter_images_aims = images_aims + r
                else:
                    brighter_images_aims = r

                # kaggle -> aims
                r = darknet(images_kaggle)
                if isinstance(darknet, ResnetGenerator):
                    darker_images_kaggle = images_kaggle + r
                else:
                    darker_images_kaggle = r


                # D_adv with REAL aims
                D_out_d = model_D_adv(images_aims)
                
                real_d = torch.FloatTensor(D_out_d.data.size()).fill_(real_label).to(device)
                fake_d = torch.FloatTensor(D_out_d.data.size()).fill_(fake_label).to(device)
                aims_d = torch.FloatTensor(D_out_d.data.size()).fill_(aims_label).to(device)
                kaggle_d = torch.FloatTensor(D_out_d.data.size()).fill_(kaggle_label).to(device)

                loss_adv_real_aims = loss_bce(D_out_d, real_d)

                # D_adv with enhanced aims
                D_out_d = model_D_adv(brighter_images_aims)
                loss_adv_brighter_aims = loss_bce(D_out_d, fake_d)

                # D_adv with REAL kaggle
                D_out_d = model_D_adv(images_kaggle)
                loss_adv_real_kaggle = loss_bce(D_out_d, real_d)

                # D_adv with enhanced kaggle
                D_out_d = model_D_adv(darker_images_kaggle)
                loss_adv_darker_kaggle = loss_bce(D_out_d, fake_d)

                # dataset discriminator
                # D_ds with REAL aims
                D_out_d = model_D_ds(images_aims)
                loss_ds_real_aims = loss_bce(D_out_d, aims_d)

                # D_ds with enhanced aims
                D_out_d = model_D_ds(brighter_images_aims)
                loss_ds_brighter_aims = loss_bce(D_out_d, aims_d)

                # D_ds with REAL kaggle
                D_out_d = model_D_ds(images_kaggle)
                loss_ds_real_kaggle = loss_bce(D_out_d, kaggle_d)

                # D_ds with enhanced kaggle
                D_out_d = model_D_ds(darker_images_kaggle)
                loss_ds_darker_kaggle = loss_bce(D_out_d, kaggle_d)


                loss = loss_adv_real_aims + loss_adv_real_kaggle + loss_adv_brighter_aims + loss_adv_darker_kaggle
                loss += loss_ds_real_aims + loss_ds_real_kaggle + loss_ds_brighter_aims + loss_ds_darker_kaggle

                loss /= 8.0

                loss_D_log = loss.item()
                
                loss.backward()
                optimizer_D.step()

                ########################################################################################################
                # LIGHTNET
                ########################################################################################################
                for p in lightnet.parameters():
                    p.requires_grad_(True)
                for p in darknet.parameters():
                    p.requires_grad_(False)
                for p in model.parameters():
                    p.requires_grad_(False)
                for p in model_D_ds.parameters():
                    p.requires_grad_(False)
                for p in model_D_adv.parameters():
                    p.requires_grad_(False)

                model.eval()
                model_D_ds.eval()
                model_D_adv.eval()
                darknet.eval()            
                lightnet.train()

                lightnet.zero_grad()
                optimizer_light.zero_grad()

                # aims -> kaggle
                if isinstance(lightnet, ResnetGenerator):
                    r = lightnet(images_aims)
                else:
                    brighter_images_aims = r
                aims_brightening = r
                brighter_images_aims = images_aims + r 

                loss_brighten_aims_ci = 0.0 if e_wgts['ciconv'] == 0 else loss_ci(brighter_images_aims, images_aims)
                loss_brighten_aims = e_wgts['tv']*loss_TV(r) + e_wgts['ssim']*torch.mean(loss_SSIM(brighter_images_aims, images_aims))\
                    + e_wgts['expz']*torch.mean(loss_exp_z(brighter_images_aims, mean_light)) + e_wgts['ciconv']*loss_brighten_aims_ci

                loss_enhancement = loss_brighten_aims
                ########################################################################################################
                # ADVERSARIAL Lightnet########################################################################
                ########################################################################################################
                # Discriminator on aims -> kaggle
                D_out_d_brighten_aims = model_D_ds(brighter_images_aims)
                loss_ds_brighten_aims = gan_criterion(D_out_d_brighten_aims, kaggle_d)

                D_out_d = model_D_adv(brighter_images_aims)
                loss_adv_brighten_aims = gan_criterion(D_out_d, real_d)
                
                loss_adv = loss_adv_brighten_aims + loss_ds_brighten_aims
                
                loss = loss_enhancement + loss_adv
                lightnet_loss = loss.item()
                loss.backward()
                optimizer_light.step()

                ########################################################################################################
                # Darknet
                ########################################################################################################
                for p in lightnet.parameters():
                    p.requires_grad_(False)
                for p in darknet.parameters():
                    p.requires_grad_(True)
                for p in model.parameters():
                    p.requires_grad_(False)
                for p in model_D_ds.parameters():
                    p.requires_grad_(False)
                for p in model_D_adv.parameters():
                    p.requires_grad_(False)

                model.eval()
                model_D_ds.eval()
                model_D_adv.eval()
                darknet.train()            
                lightnet.eval()

                darknet.zero_grad()
                optimizer_dark.zero_grad()

                # kaggle -> aims
                r = darknet(images_kaggle)
                kaggle_darkening = r
                if isinstance(darknet, ResnetGenerator):
                    darker_images_kaggle = images_kaggle + r
                else:
                    darker_images_kaggle = r

                loss_darken_kaggle_ci = 0.0 if e_wgts['ciconv'] == 0 else loss_ci(darker_images_kaggle, images_kaggle)
                loss_darken_kaggle = e_wgts['tv']*loss_TV(r) + e_wgts['ssim']*torch.mean(loss_SSIM(darker_images_kaggle, images_kaggle))\
                    + e_wgts['expz']*torch.mean(loss_exp_z(darker_images_kaggle, mean_dark)) + e_wgts['ciconv']*loss_darken_kaggle_ci\


                loss_enhancement = loss_darken_kaggle
                ########################################################################################################
                # ADVERSARIAL Darknet           ########################################################################
                ########################################################################################################
                # Discriminator on kaggle -> aims

                D_out_d = model_D_ds(darker_images_kaggle)
                loss_ds_darker_aims = gan_criterion(D_out_d, aims_d)

                D_out_d_darken_kaggle = model_D_adv(darker_images_kaggle)
                loss_adv_darker_kaggle = gan_criterion(D_out_d_darken_kaggle, real_d)

                loss_adv = loss_adv_darker_kaggle + loss_ds_darker_aims

                loss = loss_adv + loss_enhancement

                darknet_loss = loss.item()
                
                loss.backward()
                optimizer_dark.step()

            ########################################################################################################
            # Supervised training of yolo on kaggle / darker kaggle#################################################
            ########################################################################################################
            for p in lightnet.parameters():
                p.requires_grad_(False)
            for p in darknet.parameters():
                p.requires_grad_(False)
            for p in model.parameters(): # changed from only backbone
                p.requires_grad_(True)
            if args.train_style:
                for p in model_D_ds.parameters():
                    p.requires_grad_(False)
                for p in model_D_adv.parameters():
                    p.requires_grad_(False)

            model.train()
            model.zero_grad()
            lightnet.eval()
            darknet.eval()
            optimizer_model.zero_grad()

            with torch.no_grad():
                # aims -> kaggle
                r = lightnet(images_aims)
                # r = r.to(device)
                aims_brightening = r
                brighter_images_aims = torch.clamp(images_aims + r, min=0.0, max=1.0) 

                # kaggle -> aims
                r = darknet(images_kaggle)
                kaggle_darkening = r
                darker_images_kaggle = torch.clamp(images_kaggle + r, min=0.0, max=1.0) 

                # update EMA outputs for lightnet/darknet
                mean_lightnet_out = mean_lightnet_out * ema_v + (1 - ema_v) * brighter_images_aims.mean().detach().clone()
                mean_darknet_out = mean_darknet_out * ema_v + (1 - ema_v) * darker_images_kaggle.mean().detach().clone()

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

            # losses_kaggle = model(images_kaggle, instance_datas=gt_instances.clone().to(device, dtype=torch.float32), img_metas=img_metas)
            losses_darker_kaggle = model(darker_images_kaggle, instance_datas=gt_instances.clone().to(device, dtype=torch.float32), img_metas=img_metas)

            for k in loss_weights:
                losses_darker_kaggle[k] *= loss_weights[k]

            loss_yolo_kaggle_dark = sum(losses_darker_kaggle[k] for k in loss_weights)
            loss_yolo = loss_yolo_kaggle_dark

            ########################################################################################################
            # Pseudo-labeling training on aims (dark) dataset      #################################################
            ########################################################################################################
            student_images, gt_instances_pseudo, gt_instances_types, mean_psuedo_score = pseudoboxer(images_aims, brighter_images_aims, aims_img_ids)

            h, w = images_kaggle.shape[-2:]
            img_metas = [dict(ori_shape=(h, w), scale_factor=1, batch_input_shape=(h,w)) for i in range(len(images_kaggle))]
            losses_pseudolabeling = model(student_images, instance_datas=gt_instances_pseudo.clone().to(device, dtype=torch.float32), img_metas=img_metas)

            for k in loss_weights:
                losses_pseudolabeling[k] *= loss_weights[k]

            loss_yolo_pseudo = sum(losses_pseudolabeling[k] for k in loss_weights)

            # backward with both supervised loss and unsupervised loss
            loss = pseudo_tradeoff['pseudo']*loss_yolo_pseudo + pseudo_tradeoff['supervised']*loss_yolo
            
            loss.backward()
            optimizer_model.step()


            teacher_diff = 0
            # teacher_diff_n = 0
            # EMA update for the teacher
            with torch.no_grad():
                m = teacher_momentum(j)

                for param_q, param_k in zip(model.parameters(), teacher.parameters()):
                    # teacher_diff += torch.mean(torch.abs(param_k.data - param_q.data))
            #         teacher_diff_n += 1

                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            # if teacher_diff_n > 0:
            #     teacher_diff /= teacher_diff_n


            if j % 50 == 0:
                torchvision.utils.save_image(images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims.png")
                torchvision.utils.save_image(images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle.png")
                torchvision.utils.save_image(darker_images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_darker.png")
                torchvision.utils.save_image(aims_brightening.detach().cpu(), f"imgs/{wandb.run.name}/aims_brightening.png")
                torchvision.utils.save_image(brighter_images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_brighter.png")
                torchvision.utils.save_image(kaggle_darkening.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_darkening.png")
                torchvision.utils.save_image(student_images.detach().cpu(), f"imgs/{wandb.run.name}/student_images.png")

                try:
                    plot_student_pseudos(student_images, gt_instances_pseudo, gt_instances_types, save_to=f"imgs/{wandb.run.name}")
                except:
                    pass

                if len(pseudoboxer.queue) > 0:
                    up_f = torch.nn.Upsample(size=(64,64))
                    queue_images = torch.cat([up_f(x['image'].unsqueeze(0)) for x in pseudoboxer.queue], dim=0)
                    torchvision.utils.save_image(queue_images, f"imgs/{wandb.run.name}/queue.png")

            pseudo_stats = pseudoboxer.get_score_stats()

            wandb.log(step=j, data={
                'iter': i_iter,
                'loss/yolo_pseudo': loss_yolo_pseudo.item(),
                'loss/yolo': loss_yolo.item(),
                'loss/yolo_kaggle_dark': loss_yolo_kaggle_dark.item(),
                'loss/dark': 0.0 if not args.train_style else darknet_loss,
                'loss/discriminator': 0.0 if not args.train_style else loss_D_log,
                'loss/bright': 0.0 if not args.train_style else lightnet_loss,
                'pseudoboxer/min_score': pseudo_stats[0],
                'pseudoboxer/mean_score': pseudo_stats[1],
                'pseudoboxer/max_score': pseudo_stats[2],
                'pseudoboxer/queue_size': len(pseudoboxer.queue),
                'pseudoboxer/score_threshold': pseudoboxer.score_threshold,
                'style/mean_light': mean_light.item(),
                'style/mean_dark': mean_dark.item(),
                'style/mean_lightnet': mean_lightnet_out.item(),
                'style/mean_darknet': mean_darknet_out.item(),
                'teacher_momentum': teacher_momentum(j),
                # 'teacher_diff': teacher_diff.item()
            })

            if j % 250 == 0 and j > 0 and args.train_style:
                scheduler_dark.step()
                scheduler_light.step()
                scheduler_D.step()

                optim_log = {}
                for name, optim_ in [('discriminator', optimizer_D), ('dark', optimizer_dark), ('light', optimizer_light), ('yolo', optimizer_model)]:
                    optim_log[f'lr/{name}'] = optim_.param_groups[0]['lr']
                wandb.log(optim_log)

            # yolo scheduler at each step
            if isinstance(scheduler_model, optim.lr_scheduler.CosineAnnealingWarmRestarts) or isinstance(scheduler_model, optim.lr_scheduler.CosineAnnealingLR):
                scheduler_model.step()
                # scheduler_feats.step()
                wandb.log(step=j, data={
                    'lr/yolo': optimizer_model.param_groups[0]['lr']
                })
            elif j % 1000 == 0:
                scheduler_model.step()
                wandb.log(step=j, data={
                    'lr/yolo': optimizer_model.param_groups[0]['lr']
                })

            if (j % 1000 == 0 or j + 1 == len(trainloader)) and j > 0: # pre increment j
                # Validation ###################################
                map50 = evaluate(model, ds_val_aims)
                student_results = evaluate_files(gt_filename="gt_coco_format_single.json", pred_filename="eval_results_coco_format_single.json", thr=0.1)
                map50t = evaluate(teacher, ds_val_aims)
                teacher_results = evaluate_files(gt_filename="gt_coco_format_single.json", pred_filename="eval_results_coco_format_single.json", thr=0.1)
                # map50k = evaluate(teacher, ds_val_kaggle)


                wandb.log(step=j, data={
                    'val/aims/map50': map50,
                    'val/aims/sdt F2 0.3:0.8': student_results['F2 0.3:0.8'],
                    'val/aims/sdt AP 0.3:0.8': student_results['AP 0.3:0.8'],
                    'val/aims/sdt AR 0.3:0.8': student_results['AR 0.3:0.8'],
                    'val/aims/tch F2 0.3:0.8': teacher_results['F2 0.3:0.8'],
                    'val/aims/tch AP 0.3:0.8': teacher_results['AP 0.3:0.8'],
                    'val/aims/tch AR 0.3:0.8': teacher_results['AR 0.3:0.8'],
                    'val/aims/map50_teacher': map50t,
                    # 'val/kaggle/map50_teacher': map50k,
                })
                ################################################

                # copy teacher weights to student
                # model.load_state_dict(teacher.state_dict())
                # with torch.no_grad():
                #     for param_q, param_k in zip(model.parameters(), teacher.parameters()):
                #         # param_q.data.mul_(0).add_(param_k.detach().data)
                #         param_q.data = param_k.detach().data.clone()

                print('taking snapshot ...')
                torch.save(model.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_yolo_{j}' + '.pth'))
                torch.save(teacher.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_teacher_{j}' + '.pth'))
                if args.train_style:
                    torch.save(lightnet.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_light_{j}' + '.pth'))
                    torch.save(darknet.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_dark_{j}' + '.pth'))
                    torch.save(model_D_adv.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_d_adv_{j}' + '.pth'))
                    torch.save(model_D_ds.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_d_ds_{j}' + '.pth'))

            j += 1

        # if i_iter % args.save_pred_every == 0 and i_iter != 0:
        print('taking snapshot ...')
        torch.save(model.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_yolo_latest' + '.pth'))
        if args.train_style:
            torch.save(lightnet.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_light_' + "latest" + '.pth'))
            torch.save(darknet.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_dark_' + "latest" + '.pth'))
            torch.save(model_D_adv.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_d_adv_' + "latest" + '.pth'))
            torch.save(model_D_ds.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_d_ds_' + "latest" + '.pth'))

        torchvision.utils.save_image(images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_{i_iter}.png")
        torchvision.utils.save_image(images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_{i_iter}.png")
        torchvision.utils.save_image(darker_images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_darker_{i_iter}.png")
        # torchvision.utils.save_image(brighter_images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_brighter_{i_iter}.png")
        # torchvision.utils.save_image(darker_images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_darker_{i_iter}.png")
        torchvision.utils.save_image(brighter_images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_brighter_{i_iter}.png")

if __name__ == '__main__':
    main()
