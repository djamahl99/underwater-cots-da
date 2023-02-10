import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from dataset.kaggle_aims_pair_dataset_w_boxes import kaggle_aims_pair_boxed
import numpy as np
from network import *

from configs.train_config import get_arguments

import wandb

import torchvision
from network.yolo_wrapper import WrappedYOLO
from network.relighting import L_ColorInvarianceConv, Loss_bounds

from network.pseudoboxes import PseudoBoxer

from evaluate import evaluate

import PIL

def print_img_stats(img):
    print("min max", img.min(), img.max())
    img = np.clip(img, a_min=0, a_max=1)
    return img

def print_max_min(t):
    print("min max", t.min().item(), t.max().item())

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

def main():
    args = get_arguments()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda")

    cudnn.enabled = True
    cudnn.benchmark = True

    lightnet = LightNet() # aims -> kaggle
    darknet = LightNet() # kaggle -> aims
    init_weights(lightnet)
    init_weights(darknet)


    saved_state_dict_l = torch.load("snapshots/yolo/dual-discrim-epoch200-yang_light_latest.pth")
    saved_state_dict_d = torch.load("snapshots/yolo/dual-discrim-epoch200-yang_dark_latest.pth")

    # lightnet = nn.DataParallel(lightnet)
    lightnet.load_state_dict(saved_state_dict_l)
    # lightnet = lightnet.module

    # darknet = nn.DataParallel(darknet)
    darknet.load_state_dict(saved_state_dict_d)
    # darknet = darknet.module

    lightnet.train()
    lightnet.to(device)
    
    darknet.train()
    darknet.to(device)

    model_D_ds = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3) # classes = num input channels    
    saved_state_dict = torch.load("snapshots/yolo/dual-discrim-epoch200-yang_d_aims_latest.pth")
    model_D_ds.load_state_dict(saved_state_dict)
    # model_D = model_D.module

    model_D_ds.train()
    model_D_ds.to(device)

    # model_D = FCDiscriminator(num_classes=3) # classes = num input channels    
    model_D_adv = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3) # classes = num input channels    
    saved_state_dict = torch.load("snapshots/yolo/dual-discrim-epoch200-yang_d_kaggle_latest.pth")
    model_D_adv.load_state_dict(saved_state_dict)
    model_D_adv.train()
    model_D_adv.to(device)

    # yolo model
    model = WrappedYOLO()
    model.eval().to(device)

    clip_value = 0.5
    for p in model.parameters():
        p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    teacher = WrappedYOLO()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    pseudoboxer = PseudoBoxer(teacher, darknet)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    ds = kaggle_aims_pair_boxed()
    trainloader = data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )

    ds_val_aims = kaggle_aims_pair_boxed(aims_split="val.json")
    ds_val_kaggle = kaggle_aims_pair_boxed(kaggle_split="mmdet_split_val.json")

    optimizer_dark = optim.Adam(darknet.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    optimizer_light = optim.Adam(lightnet.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    optimizer_model = optim.Adam(model.backbone.parameters(), lr=args.learning_rate_yolo, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    optimizer_D = optim.Adam(list(model_D_ds.parameters()) + list(model_D_adv.parameters()), lr=args.learning_rate_D, betas=(0.9, 0.99))
    # scheduler_model = optim.lr_scheduler.ReduceLROnPlateau(optimizer_model, 'max', patience=5)

    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_model, 10, eta_min=1e-6)

    number_warmup_epochs = 3

    def warmup(current_step: int):
        if current_step > number_warmup_epochs: # cooldown
            return 0.1 ** float(current_step - number_warmup_epochs)

        return 1 / (10 ** (float(number_warmup_epochs - current_step)))
    scheduler_model = optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=warmup)

    # scheduler_model = optim.lr_scheduler.SequentialLR(optimizer_model, [warmup_scheduler, train_scheduler], [number_warmup_epochs])
    
    scheduler_dark = optim.lr_scheduler.LambdaLR(optimizer_dark, lr_lambda=lambda epoch: 0.8**(epoch))
    scheduler_light = optim.lr_scheduler.LambdaLR(optimizer_light, lr_lambda=lambda epoch: 0.8**(epoch))
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda epoch: 0.8**(epoch))

    loss_exp_z = L_exp_z(32)
    loss_TV = L_TV()
    loss_SSIM = SSIM()
    loss_bce = nn.BCEWithLogitsLoss()
    loss_ci = L_ColorInvarianceConv(invariant='W')
    loss_abs = nn.L1Loss()

    kaggle_label = 0.0
    aims_label = 1.0

    real_label = 1.0
    fake_label = 0.0

    no_crop_label = 0.0
    crop_label = 1.0

    # fixed from yang_model
    yolo_loss_weights = dict(
        loss_cls=0.22499999999999998,
        loss_bbox=0.037500000000000006,
        loss_obj=2.0999999999999996
    )

    wandb.init(
        project="DANNet YOLO DA",
        name="dual-discrim-epoch200-yang",
        config=dict(
            # lightnet_ngf=lightnet.module.ngf,
            darknet_ngf=darknet.ngf,
            discriminator=model_D_ds._get_name(),
            lr_yolo=args.learning_rate_yolo,
            # lightnet=lightnet.module._get_name(),
            darknet=darknet._get_name(),
            yolo_loss_weights=yolo_loss_weights,
            score_thres=args.teacher_score_thresh
        )
    )

    if not os.path.exists(f"imgs/{wandb.run.name}/"):
        os.makedirs(f"imgs/{wandb.run.name}/")


    wandb.watch(models=[model])

    # Validation ###################################
    map50 = evaluate(model, ds_val_aims)
    map50k = evaluate(model, ds_val_kaggle)
    wandb.log({
        'val/aims/map50': map50,
        'val/kaggle/map50': map50k,
        'lr': optimizer_model.param_groups[0]['lr']
    })
    ################################################

    # enhancement weights
    e_wgts = dict(
        tv=10,
        ssim=1,
        expz=10,
        ciconv=5.0
    )


    mean_light_ema = torch.tensor(0.5).to(device)
    ema_v = 0.99

    mean_dark_ema = torch.tensor(0.4).to(device)

    mean_lightnet_out = torch.tensor(0.5).to(device)
    mean_darknet_out = torch.tensor(0.5).to(device)

    for i_iter in range(args.num_steps):

        j = 0
        for images_kaggle, images_aims, image_ids, labels_kaggle, labels_aims, aims_img_ids in tqdm(trainloader, desc="Batch"):
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
            # r = r.to(device)
            brighter_images_aims = images_aims + r 

            # kaggle -> aims
            r = darknet(images_kaggle)
            darker_images_kaggle = images_kaggle + r

            with torch.no_grad():
                mean_lightnet_out = mean_lightnet_out * ema_v + (1 - ema_v) * brighter_images_aims.mean().detach().clone()
                mean_darknet_out = mean_darknet_out * ema_v + (1 - ema_v) * darker_images_kaggle.mean().detach().clone()


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
            r = lightnet(images_aims)
            aims_brightening = r
            brighter_images_aims = images_aims + r 

            loss_brighten_aims = e_wgts['tv']*loss_TV(r) + e_wgts['ssim']*torch.mean(loss_SSIM(brighter_images_aims, images_aims))\
                + e_wgts['expz']*torch.mean(loss_exp_z(brighter_images_aims, mean_light)) + e_wgts['ciconv']*loss_ci(brighter_images_aims, images_aims)
                    # + torch.mean(loss_SSIM(brighter_images_aims, darknet(brighter_images_aims) + brighter_images_aims))


            # kaggle -> kaggle
            # r = lightnet(images_kaggle)
            # brighter_images_kaggle = images_kaggle + r 

            # loss_brighten_kaggle = e_wgts['tv']*loss_TV(r) + e_wgts['ssim']*torch.mean(loss_SSIM(brighter_images_kaggle, images_kaggle))\
            #                 + e_wgts['expz']*torch.mean(loss_exp_z(brighter_images_kaggle, mean_light)) + e_wgts['ciconv']*loss_ci(brighter_images_kaggle, images_kaggle)

            # kaggle -> aims -> kaggle (cycle)
            # cycle_images_kaggle = images_kaggle + lightnet(images_kaggle + darknet(images_kaggle))
            # loss_brighten_cycle = 10*loss_TV(r) + 5*torch.mean(loss_SSIM(cycle_images_kaggle, images_kaggle))\
            #                 + 5*torch.mean(loss_exp_z(cycle_images_kaggle, mean_light)) + loss_bounds(brighter_images_kaggle)

            loss_enhancement = loss_brighten_aims #+ loss_brighten_kaggle #+ loss_brighten_cycle
            ########################################################################################################
            # ADVERSARIAL Lightnet########################################################################
            ########################################################################################################
            # Discriminator on aims -> kaggle
            D_out_d_brighten_aims = model_D_ds(brighter_images_aims)
            loss_ds_brighten_aims = loss_bce(D_out_d_brighten_aims, kaggle_d)

            # D_label_d_aims = torch.FloatTensor(D_out_d_brighten_aims.data.size()).fill_(aims_label).to(device) 
            # D_label_d_kaggle = torch.FloatTensor(D_out_d_brighten_aims.data.size()).fill_(kaggle_label).to(device) 

            D_out_d = model_D_adv(brighter_images_aims)
            loss_adv_brighten_aims = loss_bce(D_out_d, real_d)
            
            # Discriminator on kaggle -> kaggle
            # D_out_d_brighten_kaggle = model_D(brighter_images_kaggle)
            # loss_adv_brighten_kaggle = loss_bce(D_out_d_brighten_kaggle, D_label_d_kaggle)

            # D_out_d_cycle = model_D(cycle_images_kaggle)
            # loss_adv_brighten_cycle = loss_bce(D_out_d_cycle, D_label_d_kaggle)

            loss_adv = loss_adv_brighten_aims + loss_ds_brighten_aims #+ loss_adv_brighten_kaggle #+ loss_adv_brighten_cycle
            
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
            darker_images_kaggle = images_kaggle + r 

            loss_darken_kaggle = e_wgts['tv']*loss_TV(r) + e_wgts['ssim']*torch.mean(loss_SSIM(darker_images_kaggle, images_kaggle))\
                 + e_wgts['expz']*torch.mean(loss_exp_z(darker_images_kaggle, mean_dark)) + e_wgts['ciconv']*loss_ci(darker_images_kaggle, images_kaggle)\
                    # + torch.mean(loss_SSIM(darker_images_kaggle, lightnet(darker_images_kaggle) + darker_images_kaggle))


            # aims -> aims
            # r = darknet(images_aims)
            # darker_images_aims = images_aims + r 

            # loss_darken_aims = e_wgts['tv']*loss_TV(r) + e_wgts['ssim']*torch.mean(loss_SSIM(darker_images_aims, images_aims))\
            #                 + e_wgts['expz']*torch.mean(loss_exp_z(darker_images_aims, mean_dark)) + e_wgts['ciconv']*loss_ci(darker_images_aims, images_aims)

            # aims -> kaggle -> aims (cycle)
            # cycle_images_aims = images_aims + darknet(images_aims + lightnet(images_aims))
            # loss_darken_cycle = 10*loss_TV(r) + 5*torch.mean(loss_SSIM(cycle_images_aims, images_aims))\
            #                 + 5*torch.mean(loss_exp_z(cycle_images_aims, mean_dark)) + loss_bounds(cycle_images_aims)

            loss_enhancement = loss_darken_kaggle #+ loss_darken_aims #+ loss_darken_cycle
            ########################################################################################################
            # ADVERSARIAL Darknet           ########################################################################
            ########################################################################################################
            # Discriminator on kaggle -> aims

            D_out_d = model_D_ds(darker_images_kaggle)
            loss_ds_darker_aims = loss_bce(D_out_d, aims_d)

            D_out_d_darken_kaggle = model_D_adv(darker_images_kaggle)
            loss_adv_darker_kaggle = loss_bce(D_out_d_darken_kaggle, real_d)

            
            # Discriminator on aims -> aims
            # D_out_d_darken_aims = model_D(darker_images_aims)
            # loss_adv_darker_aims = loss_bce(D_out_d_darken_aims, D_label_d_aims)

            # discriminator on cycle
            # D_out_d_darken_cycle = model_D(cycle_images_aims)
            # loss_adv_darker_cycle = loss_bce(D_out_d_darken_cycle, D_label_d_aims)

            loss_adv = loss_adv_darker_kaggle + loss_ds_darker_aims#+ loss_adv_darker_aims #+ loss_adv_darker_cycle

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
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            for p in model_D_ds.parameters():
                p.requires_grad_(False)
            for p in model_D_adv.parameters():
                p.requires_grad_(False)

            model.train()
            model.zero_grad()
            lightnet.eval()
            
            # model_D.eval()
            darknet.eval()
            optimizer_model.zero_grad()

            with torch.no_grad():
                # kaggle -> aims
                r = darknet(images_kaggle)
                kaggle_darkening = r
                darker_images_kaggle = torch.clamp(images_kaggle + r, min=0.0, max=1.0) 

            h, w = images_kaggle.shape[-2:]
            img_metas = [dict(ori_shape=(h, w), scale_factor=1, batch_input_shape=(h,w)) for i in range(len(images_kaggle))]

            # (img_ind, labels, bbox_cx, bbox_cy, bbox_w, bbox_h)
            num_boxes = sum([0 if img_id.item() not in ds.kaggle_imgs_boxes else len(ds.kaggle_imgs_boxes[img_id.item()]['bboxes']) for img_id in image_ids])
            gt_instances = torch.zeros((num_boxes, 6), dtype=torch.float32)

            box_i = 0
            for i, img_id in enumerate(image_ids):
                if img_id.item() in ds.kaggle_imgs_boxes:
                    bboxes = ds.kaggle_imgs_boxes[img_id.item()]['bboxes']

                    for box in bboxes:
                        assert box[0::2].max() <= ds.size[1] and box[1::2].max() <= ds.size[0]
                        gt_instances[box_i, :] = torch.tensor([float(i), 1.0, *box], dtype=torch.float32)
                        box_i += 1

            losses_kaggle = model(images_kaggle, instance_datas=gt_instances.clone().to(device, dtype=torch.float32), img_metas=img_metas)
            losses_darker_kaggle = model(darker_images_kaggle, instance_datas=gt_instances.clone().to(device, dtype=torch.float32), img_metas=img_metas)

            losses_kaggle['loss_cls'] *= yolo_loss_weights['loss_cls']
            losses_kaggle['loss_bbox'] *= yolo_loss_weights['loss_bbox']
            losses_kaggle['loss_obj'] *= yolo_loss_weights['loss_obj']

            losses_darker_kaggle['loss_cls'] *= yolo_loss_weights['loss_cls']
            losses_darker_kaggle['loss_bbox'] *= yolo_loss_weights['loss_bbox']
            losses_darker_kaggle['loss_obj'] *= yolo_loss_weights['loss_obj']

            loss_yolo_kaggle = losses_kaggle['loss_cls'] + losses_kaggle['loss_obj'] + losses_kaggle['loss_bbox']
            loss_yolo_kaggle_dark = losses_darker_kaggle['loss_cls'] + losses_darker_kaggle['loss_obj'] + losses_darker_kaggle['loss_bbox']

            loss_yolo = 0.25*loss_yolo_kaggle + 0.75*loss_yolo_kaggle_dark

            loss_cls = losses_kaggle['loss_cls'] + losses_darker_kaggle['loss_cls']
            loss_obj = losses_kaggle['loss_obj'] + losses_darker_kaggle['loss_obj']
            loss_bbox = losses_kaggle['loss_bbox'] + losses_darker_kaggle['loss_bbox'] 

            ########################################################################################################
            # Pseudo-labeling training on aims (dark) dataset      #################################################
            ########################################################################################################
            student_images, gt_instances_pseudo = pseudoboxer(images_aims, brighter_images_aims, aims_img_ids)

            h, w = images_kaggle.shape[-2:]
            img_metas = [dict(ori_shape=(h, w), scale_factor=1, batch_input_shape=(h,w)) for i in range(len(images_kaggle))]
            losses_pseudolabeling = model(student_images, instance_datas=gt_instances_pseudo.clone().to(device, dtype=torch.float32), img_metas=img_metas)

            losses_pseudolabeling['loss_cls'] *= yolo_loss_weights['loss_cls']
            losses_pseudolabeling['loss_bbox'] *= yolo_loss_weights['loss_bbox']
            losses_pseudolabeling['loss_obj'] *= yolo_loss_weights['loss_obj']

            loss_yolo_pseudo = losses_pseudolabeling['loss_obj'] + losses_pseudolabeling['loss_bbox']

            # backward with both supervised loss and unsupervised loss
            loss = 0.25*loss_yolo_pseudo + 0.75*loss_yolo
            loss.backward()
            optimizer_model.step()

            p_min = lambda x: 0 if x.numel() == 0 else (x.min(), x.max())

            # EMA update for the teacher
            with torch.no_grad():
                m = args.momentum_teacher  # momentum parameter
                for param_q, param_k in zip(model.parameters(), teacher.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

            if j % 50 == 0:
                torchvision.utils.save_image(images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims.png")
                torchvision.utils.save_image(images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle.png")
                torchvision.utils.save_image(darker_images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_darker.png")
                # torchvision.utils.save_image(darkened_images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_darker.png")
                torchvision.utils.save_image(brighter_images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_brighter.png")
                # torchvision.utils.save_image(brightened_images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_brighter.png")

            # predictions
            if j == len(trainloader) - 1 or j == 0 or j % 500 == 0:
                log_images = lambda images: [wandb.Image(print_img_stats(images[i].permute(1,2,0).detach().cpu().numpy())) for i in range(images.shape[0])]
                # 
                aims_table = wandb.Table(columns=[], data=[])
                aims_table.add_column("aims", data=log_images(images_aims))
                # aims_table.add_column("darker", log_images(darker_images_aims))
                aims_table.add_column("brighter", log_images(brighter_images_aims))

                kaggle_table = wandb.Table(columns=[], data=[])

                kaggle_table.add_column("kaggle", data=log_images(images_kaggle))
                kaggle_table.add_column("darker", log_images(darker_images_kaggle))
                # kaggle_table.add_column("brighter", log_images(brighter_images_kaggle))
                kaggle_table.add_column("cots", labels_kaggle.detach().cpu().numpy())

                student_table = wandb.Table(columns=[], data=[])
                
                if student_images is not None:
                    student_table.add_column("student", log_images(student_images))
                    # crops_table.add_column("fixed", log_images(uncropped_images))

                # queue_table = wandb.Table(columns=[], data=[])
                if len(pseudoboxer.queue) > 0:
                    # concatentate to images
                    try:
                        up_f = torch.nn.Upsample(size=(64,64))
                        queue_images = torch.cat([up_f(x['image'].unsqueeze(0)) for x in pseudoboxer.queue], dim=0)
                        print("queue images", queue_images.shape)

                    
                        torchvision.utils.save_image(queue_images, f"imgs/{wandb.run.name}/queue.png")
                    except:
                        pass

                wandb.log({'aims_table': aims_table, 'kaggle_table': kaggle_table, 'student_table': student_table})
                # wandb.log({'kaggle_table': kaggle_table})

            pseudo_stats = pseudoboxer.get_score_stats()

            wandb.log({
                'iter': i_iter,
                'loss/yolo_pseudo': loss_yolo_pseudo.item(),
                'loss/yolo': loss_yolo.item(),
                'loss/yolo_kaggle': loss_yolo_kaggle.item(),
                'loss/yolo_kaggle_dark': loss_yolo_kaggle_dark.item(),
                'loss/discriminator': loss_D_log,
                'loss/cls': loss_cls.item(),
                'loss/bbox': loss_bbox.item(),
                'loss/obj': loss_obj.item(),
                'loss/dark': darknet_loss,
                'loss/bright': lightnet_loss,
                'pseudoboxer/min_score': pseudo_stats[0],
                'pseudoboxer/mean_score': pseudo_stats[1],
                'pseudoboxer/max_score': pseudo_stats[2],
                'pseudoboxer/queue_size': len(pseudoboxer.queue),
                'style/mean_light': mean_light.item(),
                'style/mean_dark': mean_dark.item(),
                'style/mean_lightnet': mean_lightnet_out.item(),
                'style/mean_darknet': mean_darknet_out.item()
            })

            if j % 250 == 0 and j > 0:
                scheduler_dark.step()
                scheduler_light.step()
                scheduler_D.step()

                optim_log = {}
                for name, optim_ in [('discriminator', optimizer_D), ('dark', optimizer_dark), ('light', optimizer_light), ('yolo', optimizer_model)]:
                    optim_log[f'lr/{name}'] = optim_.param_groups[0]['lr']
                wandb.log(optim_log)

            if j % 1000 == 0 and j > 0: # pre increment j
                # Validation ###################################
                map50 = evaluate(model, ds_val_aims)
                map50k = evaluate(model, ds_val_kaggle)

                scheduler_model.step()
                wandb.log({
                    'val/aims/map50': map50,
                    'val/kaggle/map50': map50k,
                    'lr': optimizer_model.param_groups[0]['lr']
                })
                ################################################

                print('taking snapshot ...')
                torch.save(model.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_yolo_{j}' + '.pth'))
                torch.save(lightnet.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_light_{j}' + '.pth'))
                torch.save(darknet.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_dark_{j}' + '.pth'))
                torch.save(model_D_adv.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_d_adv_{j}' + '.pth'))
                torch.save(model_D_ds.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_d_ds_{j}' + '.pth'))

            j += 1

        # if i_iter % args.save_pred_every == 0 and i_iter != 0:
        print('taking snapshot ...')
        torch.save(model.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_yolo_latest' + '.pth'))
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
