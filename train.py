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
from mmengine.structures import InstanceData

import torchvision
from network.yolo_wrapper import WrappedYOLO
from network.relighting import L_ColorInvarianceConv, Loss_bounds

from evaluate import evaluate

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def weightedMSE(D_out, D_label):
    return torch.mean((D_out - D_label).abs() ** 2)


def adjust_learning_rate(args, optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(args, optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

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

    if args.lightnet:
        lightnet = LightNet() # aims -> kaggle
    darknet = LightNet() # kaggle -> aims
    saved_state_dict = torch.load("snapshots/PSPNet/fluent-morning-24_dark_latest.pth")

    if args.lightnet:
        lightnet = nn.DataParallel(lightnet)
        # lightnet.load_state_dict(saved_state_dict)

    darknet = nn.DataParallel(darknet)
    darknet.load_state_dict(saved_state_dict)

    if args.lightnet:
        lightnet.train()
        lightnet.to(device)
    
    darknet.train()
    darknet.to(device)

    # model_D = FCDiscriminator(num_classes=3) # classes = num input channels    
    model_D = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3) # classes = num input channels    
    init_weights(model_D)
    model_D = nn.DataParallel(model_D)
    saved_state_dict = torch.load("snapshots/PSPNet/fluent-morning-24_d_latest.pth")
    # print(saved_state_dict.keys())
    # model_D.load_state_dict(saved_state_dict)
    # exit()

    model_D.train()
    model_D.to(device)

    # yolo model
    model = WrappedYOLO()
    # model = nn.DataParallel(model)
    model.eval().to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    ds = kaggle_aims_pair_boxed()
    trainloader = data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )

    ds_val_aims = kaggle_aims_pair_boxed(aims_split="val.json")

    if args.lightnet:
        optimizer_ld = optim.Adam(list(lightnet.parameters()) + list(darknet.parameters()) + list(model.parameters()), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    else:
        optimizer_ld = optim.Adam(darknet.parameters(), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay)

    optimizer_model = optim.Adam(model.parameters(), lr=args.learning_rate_yolo, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler_model = optim.lr_scheduler.ReduceLROnPlateau(optimizer_model, 'max', patience=5)
    # from yang_model
    # optimizer_model = optim.SGD(model.backbone.parameters(), lr=0.005,momentum=0.937,weight_decay=0.0005)

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()
    optimizer_ld.zero_grad()
    optimizer_model.zero_grad()

    loss_exp_z = L_exp_z(32)
    loss_TV = L_TV()
    loss_SSIM = SSIM()
    loss_bce = nn.BCEWithLogitsLoss()
    loss_ci = L_ColorInvarianceConv(invariant='W')

    kaggle_label = 0.0
    aims_label = 1.0

    yolo_loss_weights = dict(
        loss_cls=0.22499999999999998,
        loss_bbox=0.037500000000000006,
        loss_obj=2.0999999999999996
    )

    wandb.init(
        project="DANNet YOLO DA",
        config=dict(
            # lightnet_ngf=lightnet.module.ngf,
            darknet_ngf=darknet.module.ngf,
            color_invariant=loss_ci.invariant,
            discriminator=model_D.module._get_name(),
            lr_yolo=args.learning_rate_yolo,
            # lightnet=lightnet.module._get_name(),
            darknet=darknet.module._get_name(),
            yolo_loss_weights=yolo_loss_weights
        )
    )

    wandb.watch(models=[model])

    # Validation ###################################
    map50 = evaluate(model, ds_val_aims)
    wandb.log({
        'val/aims/map50': map50,
        'lr': optimizer_model.param_groups[0]['lr']
    })
    scheduler_model.step(map50)
    ################################################

    for i_iter in range(args.num_steps):
        optimizer_model.zero_grad()
        optimizer_ld.zero_grad()
        optimizer_D.zero_grad()

        j = 0
        for images_kaggle, images_aims, image_ids, labels_kaggle, labels_aims in tqdm(trainloader, desc="Batch"):
            images_aims = images_aims.to(device)
            images_kaggle = images_kaggle.to(device)
            mean_light = images_kaggle.mean()
            mean_dark = images_aims.mean()
            
            b_size = images_kaggle.shape[0]

            ########################################################################################################
            # DISCRIMINATOR 
            ########################################################################################################
            if args.lightnet:
                for p in lightnet.parameters():
                    p.requires_grad_(False)

            for p in darknet.parameters():
                p.requires_grad_(False)
            for p in model.parameters():
                p.requires_grad_(False)
            for p in model_D.parameters():
                p.requires_grad_(True)
            
            model_D.zero_grad()
            #model.zero_grad()

            if args.lightnet:
                # aims -> kaggle
                r = lightnet(images_aims)
                # r = r.to(device)
                brighter_images_aims = images_aims + r 

            # kaggle -> aims
            r = darknet(images_kaggle)
            # print("images kaggle dev", images_kaggle.device, r.device)
            darker_images_kaggle = images_kaggle + r

            # D with REAL aims
            D_out_d = model_D(images_aims)
            D_label_d = torch.FloatTensor(D_out_d.data.size()).fill_(aims_label).to(device)
            loss_adv_real_aims = loss_bce(D_out_d, D_label_d)

            if args.lightnet:
                # D with enhanced aims
                D_out_d = model_D(brighter_images_aims)
                loss_adv_brighter_aims = loss_bce(D_out_d, D_label_d)

            # D with REAL kaggle
            D_out_d = model_D(images_kaggle)
            D_label_d = torch.FloatTensor(D_out_d.data.size()).fill_(kaggle_label).to(device)
            loss_adv_real_kaggle = loss_bce(D_out_d, D_label_d)

            # D with enhanced kaggle
            D_out_d = model_D(darker_images_kaggle)
            loss_adv_darker_kaggle = loss_bce(D_out_d, D_label_d)

            if args.lightnet:
                loss = loss_adv_real_aims + loss_adv_real_kaggle + loss_adv_brighter_aims + loss_adv_darker_kaggle
            else:
                loss = loss_adv_real_aims + loss_adv_real_kaggle + loss_adv_darker_kaggle

            loss_D_log = loss.item()
            
            loss.backward()
            optimizer_D.step()

            ########################################################################################################
            # LIGHTNET & DARKNET
            ########################################################################################################
            if args.lightnet:
                for p in lightnet.parameters():
                    p.requires_grad_(True)
                lightnet.train()
            for p in darknet.parameters():
                p.requires_grad_(True)
            for p in model.parameters():
                p.requires_grad_(False)
            for p in model_D.parameters():
                p.requires_grad_(False)
            model.eval()
            model_D.eval()
            darknet.train()            

            if args.lightnet:
                lightnet.zero_grad()
            darknet.zero_grad()
            # model.zero_grad()
            optimizer_ld.zero_grad()

            if args.lightnet:
                # aims -> kaggle
                r = lightnet(images_aims)
                aims_brightening = r
                brighter_images_aims = images_aims + r 

                loss_ci_aims_brighten = loss_ci(brighter_images_aims, images_aims)
                loss_brighten_aims = 10*loss_TV(r)+torch.mean(loss_SSIM(brighter_images_aims, images_aims))\
                    + 5*torch.mean(loss_exp_z(brighter_images_aims, mean_light)) + 0.0*loss_ci_aims_brighten

            # kaggle -> aims
            r = darknet(images_kaggle)
            kaggle_darkening = r
            darker_images_kaggle = images_kaggle + r 

            loss_ci_kaggle_darken = loss_ci(darker_images_kaggle, images_aims)
            loss_darken_kaggle = 10*loss_TV(r)+torch.mean(loss_SSIM(darker_images_kaggle, images_aims))\
                 + 5*torch.mean(loss_exp_z(darker_images_kaggle, mean_dark)) + 0.0*loss_ci_kaggle_darken

            if args.lightnet:
                # kaggle -> kaggle
                r = lightnet(images_kaggle)
                brighter_images_kaggle = images_kaggle + r 

                loss_ci_kaggle_brighten = loss_ci(brighter_images_kaggle, images_kaggle)
                loss_brighten_kaggle = 10*loss_TV(r) + 5*torch.mean(loss_SSIM(brighter_images_kaggle, images_kaggle))\
                                + 5*torch.mean(loss_exp_z(brighter_images_kaggle, mean_light)) + 0.0*loss_ci_kaggle_brighten

            # aims -> aims
            r = darknet(images_aims)
            darker_images_aims = images_aims + r 

            loss_ci_aims_darken = loss_ci(darker_images_aims, images_aims)
            loss_darken_aims = 10*loss_TV(r) + 5*torch.mean(loss_SSIM(darker_images_aims, images_aims))\
                            + 5*torch.mean(loss_exp_z(darker_images_aims, mean_dark)) + 0.0*loss_ci_aims_darken

            if args.lightnet:
                loss_enhancement = loss_brighten_aims + loss_darken_kaggle + loss_brighten_kaggle + loss_darken_aims
            else:
                loss_enhancement = loss_darken_kaggle + loss_darken_aims
            ########################################################################################################
            # ADVERSARIAL Lightnet & Darknet########################################################################
            ########################################################################################################

            if args.lightnet:
                # Discriminator on aims -> kaggle
                D_out_d_brighten_aims = model_D(brighter_images_aims)

                D_label_d_aims = torch.FloatTensor(D_out_d_brighten_aims.data.size()).fill_(aims_label).to(device) 
                D_label_d_kaggle = torch.FloatTensor(D_out_d_brighten_aims.data.size()).fill_(kaggle_label).to(device) 

                loss_adv_brighten_aims = loss_bce(D_out_d_brighten_aims, D_label_d_kaggle)
                
                # Discriminator on kaggle -> kaggle
                D_out_d_brighten_kaggle = model_D(brighter_images_kaggle)
                loss_adv_brighten_kaggle = loss_bce(D_out_d_brighten_kaggle, D_label_d_kaggle)

            # Discriminator on kaggle -> aims
            D_out_d_darken_kaggle = model_D(darker_images_kaggle)

            if not args.lightnet:
                D_label_d_aims = torch.FloatTensor(D_out_d_darken_kaggle.data.size()).fill_(aims_label).to(device) 

            loss_adv_darker_aims = loss_bce(D_out_d_darken_kaggle, D_label_d_aims)
            
            # Discriminator on aims -> aims
            D_out_d_darken_aims = model_D(darker_images_aims)
            loss_adv_darker_kaggle = loss_bce(D_out_d_darken_aims, D_label_d_aims)

            if args.lightnet:
                loss_adv = loss_adv_brighten_aims + loss_adv_brighten_kaggle + loss_adv_darker_kaggle + loss_adv_darker_aims
            else:
                loss_adv = loss_adv_darker_kaggle + loss_adv_darker_aims

            loss = loss_adv + loss_enhancement
            
            loss.backward()
            optimizer_ld.step()

            ########################################################################################################
            # Train yolo on kaggle / darker kaggle################################################################
            ########################################################################################################
            if args.lightnet:
                for p in lightnet.parameters():
                    p.requires_grad_(False)
                lightnet.eval()
            for p in darknet.parameters():
                p.requires_grad_(False)
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            for p in model_D.parameters():
                p.requires_grad_(False)

            model.train()
            model.zero_grad()
            
            model_D.eval()
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

            # losses_kaggle = model(images_kaggle, instance_datas=gt_instances.clone().to(device, dtype=torch.float32), img_metas=img_metas)
            losses_kaggle = {'loss_cls': 0, 'loss_obj': 0, 'loss_bbox': 0}

            losses_darker_kaggle = model(darker_images_kaggle, instance_datas=gt_instances.clone().to(device, dtype=torch.float32), img_metas=img_metas)

            losses_kaggle['loss_cls'] *= yolo_loss_weights['loss_cls']
            losses_kaggle['loss_bbox'] *= yolo_loss_weights['loss_bbox']
            losses_kaggle['loss_obj'] *= yolo_loss_weights['loss_obj']

            losses_darker_kaggle['loss_cls'] *= yolo_loss_weights['loss_cls']
            losses_darker_kaggle['loss_bbox'] *= yolo_loss_weights['loss_bbox']
            losses_darker_kaggle['loss_obj'] *= yolo_loss_weights['loss_obj']

            loss_yolo_kaggle = losses_kaggle['loss_cls'] + losses_kaggle['loss_obj'] + losses_kaggle['loss_bbox']
            loss_yolo_kaggle = torch.tensor(0)
            loss_yolo_kaggle_dark = losses_darker_kaggle['loss_cls'] + losses_darker_kaggle['loss_obj'] + losses_darker_kaggle['loss_bbox']

            loss_yolo = loss_yolo_kaggle + loss_yolo_kaggle_dark

            loss_cls = losses_kaggle['loss_cls'] + losses_darker_kaggle['loss_cls']
            loss_obj = losses_kaggle['loss_obj'] + losses_darker_kaggle['loss_obj']
            loss_bbox = losses_kaggle['loss_bbox'] + losses_darker_kaggle['loss_bbox'] 
            ###########################################################################################

            loss = loss_yolo
            
            loss.backward()
            optimizer_model.step()

            # boxes, scores = model.forward_pred_no_grad(images_kaggle[0].unsqueeze(0))
            # boxes = boxes.flatten(0,1)
            # scores = scores.flatten(0,1)

            # print("boxes sores",  boxes, scores)
            # p_min = lambda x: 0 if x.numel() == 0 else (x.min(), x.max())
            # print("gt_instances 374", gt_instances, p_min(gt_instances))
            # print("gt_instances_orig 375", gt_instances_orig)

            # if boxes.numel() > 0 and gt_instances_orig.numel() > 0:
            #     import matplotlib.pyplot as plt
            #     from matplotlib.patches import Rectangle

            #     #### PLOTTING
            #     plt.figure(figsize=(20,20))
            #     #define Matplotlib figure and axis
            #     # fig, ax = plt.subplots()
            #     ax = plt.subplot(1,1,1)

            #     ax.imshow(images_kaggle[0].permute(1, 2, 0).detach().cpu().numpy())

            #     line_styles = ['-', '--', '-.', ':']

            #     box_num = 0

            #     for i, bbox in enumerate(boxes):
            #         score = scores[i]
            #         print("ibox", i, bbox, score)
                    

            #         bbox = xyxy2matplotlibxywh(bbox)

            #         label = "pred"

            #         #add rectangle to plot
            #         ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], alpha=1, fill=False, lw=1, label=label, color=[c/255 for c in PALETTE[box_num]]))
            #         box_num += 1

            #     for i in range(gt_instances_orig.shape[0]):
            #         bbox = gt_instances_orig[i, 2:]
            #         bbox = xyxy2matplotlibxywh(bbox)
            #         label = "GT"
            #         #add rectangle to plot
            #         ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], alpha=1, fill=False, lw=1, label=label, color=[c/255 for c in PALETTE[box_num]]))
            #         box_num += 1
                
            #     plt.legend()
            #     plt.savefig("inference.png")

            #     exit()

            # predictions
            if j == len(trainloader) - 1 or j == 0 or j % 1000 == 0:

                #print_img_stats = lambda image: print("min max", image.min(), image.max())

                def print_img_stats(img):

                    print("min max", img.min(), img.max())
                    img = np.clip(img, a_min=0, a_max=1)
                    return img

                log_images = lambda images: [wandb.Image(print_img_stats(images[i].permute(1,2,0).detach().cpu().numpy())) for i in range(images.shape[0])]
                # 
                # aims_table = wandb.Table(columns=[], data=[])
                # aims_table.add_column("aims", data=log_images(images_aims))
                # aims_table.add_column("darker", log_images(darker_images_aims))
                # if args.lightnet:
                    # aims_table.add_column("brighter", log_images(brighter_images_aims))
                # aims_table.add_column("cots", labels_aims.detach().cpu().numpy())

                kaggle_table = wandb.Table(columns=[], data=[])

                kaggle_table.add_column("kaggle", data=log_images(images_kaggle))
                kaggle_table.add_column("darker", log_images(darker_images_kaggle))
                if args.lightnet:
                    kaggle_table.add_column("brighter", log_images(brighter_images_kaggle))
                # kaggle_table.add_column("D(enhanced) aims=1", D_out_d_kaggle.mean(dim=(1,2,3)).detach().cpu().numpy())
                kaggle_table.add_column("cots", labels_kaggle.detach().cpu().numpy())
                # kaggle_table.add_column("P(cots)", pred_labels.sum(1).detach().cpu().numpy())

                # wandb.log({'aims_table': aims_table, 'kaggle_table': kaggle_table})
                wandb.log({'kaggle_table': kaggle_table})

            j += 1

            wandb.log({
                'iter': i_iter,
                'loss/yolo': loss_yolo.item(),
                'loss/yolo_kaggle': 0 if type(loss_yolo_kaggle) != torch.tensor else loss_yolo_kaggle.item(),
                'loss/yolo_kaggle_dark': loss_yolo_kaggle_dark.item(),
                'loss/enhance': loss_enhancement.item(),
                'loss/discriminator': loss_D_log,
                'loss/cls': loss_cls.item(),
                'loss/bbox': loss_bbox.item(),
                'loss/obj': loss_obj.item(),
                'loss/adversarial': loss_adv.item()
            })

            if j % 1000 == 0:
                # Validation ###################################
                map50 = evaluate(model, ds_val_aims)
                scheduler_model.step(map50)
                wandb.log({
                    'val/aims/map50': map50,
                    'lr': optimizer_model.param_groups[0]['lr']
                })
                ################################################

        # if i_iter % args.save_pred_every == 0 and i_iter != 0:
        print('taking snapshot ...')
        torch.save(model.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_yolo_latest' + '.pth'))
        if args.lightnet:
            torch.save(lightnet.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_light_' + "latest" + '.pth'))
        torch.save(darknet.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_dark_' + "latest" + '.pth'))
        torch.save(model_D.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_d_' + "latest" + '.pth'))
        # torch.save(model_D2.state_dict(), os.path.join(args.snapshot_dir, 'dannet_d2_' + str(i_iter) + '.pth'))


        if not os.path.exists(f"imgs/{wandb.run.name}/"):
            os.makedirs(f"imgs/{wandb.run.name}/")

        torchvision.utils.save_image(images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_{i_iter}.png")
        torchvision.utils.save_image(images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_{i_iter}.png")
        torchvision.utils.save_image(darker_images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_darker_{i_iter}.png")
        if args.lightnet:
            torchvision.utils.save_image(brighter_images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_brighter_{i_iter}.png")
        torchvision.utils.save_image(darker_images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_darker_{i_iter}.png")
        if args.lightnet:
            torchvision.utils.save_image(brighter_images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_brighter_{i_iter}.png")

if __name__ == '__main__':
    main()
