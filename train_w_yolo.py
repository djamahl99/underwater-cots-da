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

from network.deeprelight_networks import define_G, VGGLoss

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

def main():
    args = get_arguments()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device("cuda")

    cudnn.enabled = True
    cudnn.benchmark = True

    lightnet = LightNet(8, 1) # aims -> kaggle
    darknet = LightNet(8, 1) # kaggle -> aims
    # saved_state_dict = torch.load("snapshots/exalted-snow-144_light_latest.pth")

    # lightnet = nn.DataParallel(lightnet, device_ids=[0], output_device=0)
    lightnet = nn.DataParallel(lightnet)
    darknet = nn.DataParallel(darknet)

    # lightnet.load_state_dict(saved_state_dict)
    # darknet.load_state_dict(saved_state_dict)

    lightnet.train()
    darknet.train()
    lightnet.to(device)
    darknet.to(device)

    model_D = FCDiscriminator(num_classes=3) # classes = num input channels    
    model_D = nn.DataParallel(model_D)
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

    optimizer = optim.Adam(list(lightnet.parameters()) + list(darknet.parameters()) + list(model.parameters()), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    loss_exp_z = L_exp_z(32)
    loss_TV = L_TV()
    loss_SSIM = SSIM()
    loss_bce = nn.BCELoss()
    loss_ci = L_ColorInvarianceConv(invariant='W')
    loss_bounds = Loss_bounds()

    kaggle_label = 0.0
    aims_label = 1.0

    wandb.init(
        project="DANNet YOLO DA",
        config=dict(
            lightnet_ngf=lightnet.module.ngf,
            darknet_ngf=darknet.module.ngf,
            color_invariant=loss_ci.invariant,
            discriminator=model_D.module._get_name(),
            lightnet=lightnet.module._get_name(),
            darknet=darknet.module._get_name()
        )
    )

    wandb.watch(models=[lightnet, darknet, model_D, model])

    for i_iter in range(args.num_steps):
        optimizer.zero_grad()
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

            loss = loss_adv_real_aims + loss_adv_real_kaggle + loss_adv_brighter_aims + loss_adv_darker_kaggle

            loss_D_log = loss.item()
            
            loss.backward()
            optimizer_D.step()

            ########################################################################################################
            # LIGHTNET & DARKNET
            ########################################################################################################
            for p in lightnet.parameters():
                p.requires_grad_(True)
            for p in darknet.parameters():
                p.requires_grad_(True)
            for p in model.parameters():
                p.requires_grad_(True)
            for p in model_D.parameters():
                p.requires_grad_(False)

            lightnet.zero_grad()
            darknet.zero_grad()
            # model.zero_grad()

            # aims -> kaggle
            r = lightnet(images_aims)
            aims_brightening = r
            brighter_images_aims = images_aims + r 

            loss_ci_aims_brighten = loss_ci(brighter_images_aims, images_aims)
            loss_brighten_aims = 10*loss_TV(r)+torch.mean(loss_SSIM(brighter_images_aims, images_aims))\
                 + 5*torch.mean(loss_exp_z(brighter_images_aims, mean_light)) + loss_ci_aims_brighten

            # kaggle -> aims
            r = darknet(images_kaggle)
            kaggle_darkening = r
            darker_images_kaggle = images_kaggle + r 

            loss_ci_kaggle_darken = loss_ci(darker_images_kaggle, images_aims)
            loss_darken_kaggle = 10*loss_TV(r)+torch.mean(loss_SSIM(darker_images_kaggle, images_aims))\
                 + 5*torch.mean(loss_exp_z(darker_images_kaggle, mean_dark)) + loss_ci_kaggle_darken

            # kaggle -> kaggle
            r = lightnet(images_kaggle)
            brighter_images_kaggle = images_kaggle + r 

            loss_ci_kaggle_brighten = loss_ci(brighter_images_kaggle, images_kaggle)
            loss_brighten_kaggle = 10*loss_TV(r) + 5*torch.mean(loss_SSIM(brighter_images_kaggle, images_kaggle))\
                            + 5*torch.mean(loss_exp_z(brighter_images_kaggle, mean_light)) + loss_ci_kaggle_brighten

            # aims -> aims
            r = darknet(images_aims)
            darker_images_aims = images_aims + r 

            loss_ci_aims_darken = loss_ci(darker_images_aims, images_aims)
            loss_darken_aims = 10*loss_TV(r) + 5*torch.mean(loss_SSIM(darker_images_aims, images_aims))\
                            + 5*torch.mean(loss_exp_z(darker_images_aims, mean_dark)) + loss_ci_aims_darken

            loss_enhancement = loss_brighten_aims + loss_darken_kaggle + loss_brighten_kaggle + loss_darken_aims

            ########################################################################################################
            # ADVERSARIAL Lightnet & Darknet########################################################################
            ########################################################################################################

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
            loss_adv_darker_aims = loss_bce(D_out_d_darken_kaggle, D_label_d_aims)
            
            # Discriminator on aims -> aims
            D_out_d_darken_aims = model_D(darker_images_aims)
            loss_adv_darker_kaggle = loss_bce(D_out_d_darken_aims, D_label_d_aims)

            loss_adv = loss_adv_brighten_aims + loss_adv_brighten_kaggle + loss_adv_darker_kaggle + loss_adv_darker_aims

            ########################################################################################################
            # Train yolo on kaggle / darker kaggle################################################################
            ########################################################################################################
            h, w = images_kaggle.shape[-2:]
            img_metas = [dict(ori_shape=(h, w), scale_factor=1, batch_input_shape=(h,w)) for i in range(len(images_kaggle))]

            # (img_ind, labels, bbox_cx, bbox_cy, bbox_w, bbox_h)
            gt_instances = []

            for i, img_id in enumerate(image_ids):
                if img_id.item() in ds.kaggle_imgs_boxes:
                    bboxes = ds.kaggle_imgs_boxes[img_id.item()]['bboxes']

                    for box in bboxes:
                        gt_instances.append([i, 1, *box])

            gt_instances = torch.tensor(gt_instances).to(device)

            losses_kaggle = model(images_kaggle, instance_datas=gt_instances, img_metas=img_metas)
            losses_darker_kaggle = model(darker_images_kaggle, instance_datas=gt_instances, img_metas=img_metas)

            loss_yolo_kaggle = losses_kaggle['loss_cls'] + losses_kaggle['loss_obj'] + losses_kaggle['loss_bbox']
            loss_yolo_kaggle_dark = losses_darker_kaggle['loss_cls'] + losses_darker_kaggle['loss_obj'] + losses_darker_kaggle['loss_bbox']

            loss_yolo = loss_yolo_kaggle + loss_yolo_kaggle_dark

            loss_cls = losses_kaggle['loss_cls'] + losses_darker_kaggle['loss_cls']
            loss_obj = losses_kaggle['loss_obj'] + losses_darker_kaggle['loss_obj']
            loss_bbox = losses_kaggle['loss_bbox'] + losses_darker_kaggle['loss_bbox'] 
            ###########################################################################################

            loss = loss_yolo + loss_adv + loss_enhancement
            
            loss.backward()
            optimizer.step()

            # predictions
            if j == len(trainloader) - 1 or j == 0 or j % 50 == 0:
                aims_table = wandb.Table(columns=[], data=[])

                #print_img_stats = lambda image: print("min max", image.min(), image.max())

                def print_img_stats(img):

                    print("min max", img.min(), img.max())
                    img = np.clip(img, a_min=0, a_max=1)
                    return img

                log_images = lambda images: [wandb.Image(print_img_stats(images[i].permute(1,2,0).detach().cpu().numpy())) for i in range(images.shape[0])]

                aims_table.add_column("aims", data=log_images(images_aims))
                aims_table.add_column("darker", log_images(darker_images_aims))
                aims_table.add_column("brighter", log_images(brighter_images_aims))
                aims_table.add_column("cots", labels_aims.detach().cpu().numpy())

                kaggle_table = wandb.Table(columns=[], data=[])

                kaggle_table.add_column("kaggle", data=log_images(images_kaggle))
                kaggle_table.add_column("darker", log_images(darker_images_kaggle))
                kaggle_table.add_column("brighter", log_images(brighter_images_kaggle))
                # kaggle_table.add_column("D(enhanced) aims=1", D_out_d_kaggle.mean(dim=(1,2,3)).detach().cpu().numpy())
                kaggle_table.add_column("cots", labels_kaggle.detach().cpu().numpy())
                # kaggle_table.add_column("P(cots)", pred_labels.sum(1).detach().cpu().numpy())

                wandb.log({'aims_table': aims_table, 'kaggle_table': kaggle_table})

            j += 1

            wandb.log({
                'iter': i_iter,
                # 'num_steps': args.num_steps,
                'loss/yolo': loss_yolo.item(),
                'loss/yolo+lightnet+darknet': loss.item(),
                'loss/enhance': loss_enhancement.item(),
                'loss/discriminatior': loss_D_log,
                'loss/cls': loss_cls.item(),
                'loss/bbox': loss_bbox.item(),
                'loss/obj': loss_obj.item(),
                'loss/adversarial': loss_adv.item()
            })

        # if i_iter % args.save_pred_every == 0 and i_iter != 0:
        print('taking snapshot ...')
        torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'cots_classifier' + '.pth'))
        torch.save(lightnet.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_light_' + "latest" + '.pth'))
        torch.save(model_D.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_d_' + "latest" + '.pth'))
        # torch.save(model_D2.state_dict(), os.path.join(args.snapshot_dir, 'dannet_d2_' + str(i_iter) + '.pth'))

        if not os.path.exists(f"imgs/{wandb.run.name}/"):
            os.makedirs(f"imgs/{wandb.run.name}/")

        torchvision.utils.save_image(images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_{i_iter}.png")
        torchvision.utils.save_image(images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_{i_iter}.png")
        torchvision.utils.save_image(darker_images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_darker_{i_iter}.png")
        torchvision.utils.save_image(brighter_images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_brighter_{i_iter}.png")
        torchvision.utils.save_image(darker_images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_darker_{i_iter}.png")
        torchvision.utils.save_image(brighter_images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_brighter_{i_iter}.png")

if __name__ == '__main__':
    main()
