import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from dataset.kaggle_aims_pair_dataset import kaggle_aims_pair
import numpy as np
from network import *

from configs.train_config import get_arguments

import wandb

import torchvision
from network.discriminator import ResNet18Discriminator, YOLODiscriminator

from network.relighting import GammaLightNet, L_ColorInvarianceConv, L_grayscale, Loss_bounds


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
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda")
    device2 = torch.device("cuda")

    cudnn.enabled = True
    cudnn.benchmark = True

    generator = "lightnet" # lightnet
    discriminator = "fc" # yolo
    r_coeff = 1.0

    lightnet = LightNet(ngf=32)
    # saved_state_dict = torch.load("dannet_deeplab_light.pth")
    init_weights(lightnet)

    # saved_state_dict = torch.load("snapshots/yolo/enchanting-lamp-191_light_latest.pth")

    # from network.relighting import ResNetColorCorrector2
    # lightnet = ResNetColorCorrector2()

    lightnet = nn.DataParallel(lightnet)
    # lightnet.load_state_dict(saved_state_dict)
    #if generator == "lightnet":
    #    lightnet.load_state_dict(saved_state_dict)
    lightnet.train()
    lightnet.to(device)

    # if discriminator == "fc":
    #     model_D = FCDiscriminator(num_classes=3) # classes = num input channels
    # else:
    #     model_D = YOLODiscriminator()

    model_D = NLayerDiscriminator(input_nc=3, ndf=32)
    model_D = nn.DataParallel(model_D)
    # model_D.load_state_dict(torch.load("snapshots/yolo/enchanting-lamp-191_d_latest.pth"))
    # 
    # if discriminator == "yolo":
    #     #model_D.load_state_dict(saved_state_dict)
    #     saved_state_dict = torch.load("snapshots/wobbly-puddle-108_d_1.pth")
    #     model_D.load_state_dict(saved_state_dict)
    # model_D = ResNet18Discriminator()
    #model_D = nn.DataParallel(model_D)
    # model_D.backbone.eval()
    
    model_D.train()
    model_D.to(device2)

    ###print(model_D.module)
    
    # if running classification
    # model = FCDiscriminator(num_classes=3, patched=False, ndf=2)
    # model = nn.DataParallel(model)
    # model.train()
    # model.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    ds = kaggle_aims_pair()
    trainloader = data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    trainloader_iter = enumerate(trainloader)

    optimizer = optim.Adam(list(lightnet.parameters()), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    optimizer.zero_grad()

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.1**(epoch / 10))

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda epoch: 0.1**(epoch / 10))

    loss_exp_z = L_exp_z(32)
    loss_TV = L_TV()
    loss_SSIM = SSIM()
    loss_bce = nn.BCEWithLogitsLoss()
    loss_mse = nn.MSELoss()
    loss_l1 = nn.L1Loss()
    loss_ci = L_ColorInvarianceConv(invariant='W')
    loss_gray = L_grayscale()
    # loss_bounds = Loss_bounds()
    # loss_vgg = VGGLoss()

    kaggle_label = 0.0
    aims_label = 1.0
    fake_label = 0.0
    real_label = 1.0

    #r_coeff = 1.0

    wandb.init(
        project="UNet (DANNet) adversarial AIMS -> Kaggle",
        config=dict(
            r_coeff=r_coeff,
            # lightnet_ngf=lightnet.module.ngf,
            color_invariant=loss_ci.invariant,
            discriminator=model_D.module._get_name(),
            lightnet=lightnet.module._get_name(),
            # classifier=model.module._get_name()
        )
    )

    wandb.watch(models=[lightnet])

    mean_light_ema = torch.tensor(0.5).to(device)
    ema_v = 0.99

    mean_dark_ema = torch.tensor(0.5).to(device)

    # enhancement weights
    e_wgts = dict(
        tv=10,
        ssim=1,
        expz=1,
        ciconv=2
    )
    adv_coeff = 1.0

    for i_iter in range(args.num_steps):
        optimizer.zero_grad()
        optimizer_D.zero_grad()

        j = 0
        for images_kaggle, images_aims, labels_kaggle, labels_aims in tqdm(trainloader, desc="Batch"):
            labels_kaggle = labels_kaggle.to(device)
            #labels_aims = labels_aims.to(device)

            images_aims = images_aims.to(device)
            with torch.no_grad():
                mean_r = images_kaggle[:, 0].mean()
                mean_g = images_kaggle[:, 1].mean()
                mean_b = images_kaggle[:, 2].mean()

                # mean_light = images_kaggle.mean()
                mean_light = (mean_r + mean_g + mean_b) / 3.0
                mean_light_ema = mean_light_ema * ema_v + (1 - ema_v) * mean_light
            # to loss
            mean_light = mean_light_ema
            
            images_kaggle = images_kaggle.to(device)

            b_size = images_kaggle.shape[0]

            ########################################################################################################
            # DISCRIMINATOR 
            ########################################################################################################
            for p in lightnet.parameters():
                p.requires_grad_(False)
            for p in model_D.parameters():
                p.requires_grad_(True)
            
            model_D.zero_grad()
            #model.zero_grad()

            # enhance aims images
            r = lightnet(images_aims)
            if generator == "lightnet":
                enhanced_images_aims = images_aims + r * r_coeff
            else: 
                enhanced_images_aims = r
            # enhanced_images_aims = torch.clamp(enhanced_images_aims, 0.0, 1.0)

            # enhance kaggle images
            #r = lightnet(images_kaggle)
            #if generator == "lightnet":
            #    enhanced_images_kaggle = images_kaggle + r * r_coeff
            #else:
            #    enhanced_images_kaggle = r
            # enhanced_images_kaggle = torch.clamp(enhanced_images_kaggle, 0.0, 1.0)

            # D with REAL aims
            D_out_d = model_D(images_aims)
            # D_label_d = torch.tensor([[aims_label]], dtype=torch.float).repeat((b_size, 1)).to(device)
            D_label_d = torch.FloatTensor(D_out_d.data.size()).fill_(aims_label).to(device2) # use kaggle as want lightnet to produce kaggle style
            loss_adv_real_aims = loss_bce(D_out_d, D_label_d)

            # D with enhanced aims
            D_out_d = model_D(enhanced_images_aims)
            loss_adv_enhanced_aims = loss_bce(D_out_d, D_label_d)

            # D with REAL kaggle
            D_out_d = model_D(images_kaggle)
            D_label_d = torch.FloatTensor(D_out_d.data.size()).fill_(kaggle_label).to(device2) # use kaggle as want lightnet to produce kaggle style
            loss_adv_real_kaggle = loss_bce(D_out_d, D_label_d)

            # D with enhanced kaggle (treat as real) - will make discriminator know about noisy kaggle
            #D_out_d = model_D(enhanced_images_kaggle)
            #loss_adv_enhanced_kaggle = loss_bce(D_out_d, D_label_d)

            loss = loss_adv_real_aims + loss_adv_real_kaggle + loss_adv_enhanced_aims #+ loss_adv_enhanced_kaggle
            loss = loss / args.iter_size

            loss_D_log = loss.item()
            
            loss.backward()
            optimizer_D.step()

            ########################################################################################################
            # LIGHTNET
            ########################################################################################################
            for p in lightnet.parameters():
                p.requires_grad_(True)
            for p in model_D.parameters():
                p.requires_grad_(False)

            lightnet.zero_grad()
            # model.zero_grad()

            # aims -> enhanced
            r = lightnet(images_aims)
            enhancement_aims = r
            if generator == "lightnet":
                enhanced_images_aims = images_aims + r * r_coeff
            else:
                enhanced_images_aims = r
            # loss_enhance_bounds_aims = loss_bounds(enhanced_images_aims) # bounds loss
            loss_ci_aims = loss_ci(enhanced_images_aims, images_aims)
            # loss_ci_aims = torch.tensor(0)
            loss_enhance_aims = e_wgts['tv']*loss_TV(r) + e_wgts['ssim']*torch.mean(loss_SSIM(enhanced_images_aims, images_aims))\
                 + e_wgts['expz']*torch.mean(loss_exp_z(enhanced_images_aims, mean_light))\
                    + e_wgts['ciconv']*loss_ci_aims
                    # + loss_gray(enhanced_images_aims, images_aims)
                    # + loss_enhance_bounds_aims

            # kaggle -> enhanced
            r = lightnet(images_kaggle)
            if generator == "lightnet":
                enhanced_images_kaggle = images_kaggle + r * r_coeff
            else:
                enhanced_images_kaggle = r

            zeros_r = torch.FloatTensor(r.data.size()).fill_(0).to(device2)
            
            # loss_enhance_bounds_kaggle = loss_bounds(enhanced_images_kaggle) # bounds loss
            loss_ci_kaggle = loss_ci(enhanced_images_kaggle, images_kaggle)
            # loss_ci_kaggle = torch.tensor(0)
            loss_enhance_kaggle = e_wgts['tv']*loss_TV(r) + e_wgts['ssim']*torch.mean(loss_SSIM(enhanced_images_kaggle, images_kaggle))\
                            + e_wgts['expz']*torch.mean(loss_exp_z(enhanced_images_kaggle, mean_light))\
                            + e_wgts['ciconv']*loss_ci_kaggle
                            # + loss_gray(enhanced_images_kaggle, images_kaggle)
                                # + loss_l1(r, zeros_r)

            # Discriminator on enhanced kaggle
            D_out_d = model_D(enhanced_images_kaggle)
            D_out_d_kaggle = D_out_d
            # scalar
            # D_label_d = torch.tensor([[kaggle_label]], dtype=torch.float).repeat((b_size, 1)).to(device)
            # patched
            D_label_d = torch.FloatTensor(D_out_d.data.size()).fill_(kaggle_label).to(device2) # use kaggle as want lightnet to produce kaggle style
            loss_adv_enhanced_kaggle = loss_bce(D_out_d, D_label_d)
            
            # Discriminator on enhanced aims
            D_out_d = model_D(enhanced_images_aims) # use same D_label-d
            loss_adv_enhanced_aims = loss_bce(D_out_d, D_label_d)

            ###### classifier task ####################################################################
            # labels_kaggle = labels_kaggle.unsqueeze(1)
            # labels_aims = labels_aims.unsqueeze(1)
            # orig kaggle
            # pred_labels = model(images_kaggle)
            # labels_kaggle_ = labels_kaggle.reshape_as(pred_labels)
            # loss_classify_kaggle = loss_bce(pred_labels, labels_kaggle_)

            # orig aims
            #pred_labels = model(images_aims)
            #loss_classify_aims = loss_bce(pred_labels, labels_aims)

            # # enhanced aims
            # pred_labels = model(enhanced_images_aims)
            # loss_classify_aims_enhanced = loss_bce(pred_labels, labels_aims)

            # enhanced kaggle
            # pred_labels = model(enhanced_images_kaggle)
            # loss_classify_kaggle_enhanced = loss_bce(pred_labels, labels_kaggle)

            # labels_kaggle = labels_kaggle.squeeze(1)
            # labels_aims = labels_aims.squeeze(1)

            # loss_classifier = loss_classify_kaggle + loss_classify_kaggle_enhanced
            #loss_classifier = loss_classify_kaggle + loss_classify_aims + loss_classify_aims_enhanced + loss_classify_kaggle_enhanced
            ###########################################################################################


            loss = adv_coeff * (loss_adv_enhanced_kaggle + loss_adv_enhanced_aims) + loss_enhance_aims + loss_enhance_kaggle #+ 5*loss_classifier
            loss = loss / args.iter_size

            loss_enhance = loss_enhance_aims.item() + loss_enhance_kaggle.item()

            loss_lightnet_log = loss.item()

            # loss_enhance_bounds =  loss_enhance_bounds_aims.item() + loss_enhance_bounds_kaggle.item()
            loss_enhance_ci = loss_ci_aims.item() + loss_ci_kaggle.item()

            
            loss.backward()
            optimizer.step()

            # predictions
            if j == len(trainloader) - 1 or j == 0 or j % 500 == 0:
                aims_table = wandb.Table(columns=[], data=[])

                #print_img_stats = lambda image: print("min max", image.min(), image.max())

                def print_img_stats(img):

                    print("min max", img.min(), img.max())
                    img = np.clip(img, a_min=0, a_max=1)
                    return img

                log_images = lambda images: [wandb.Image(print_img_stats(images[i].permute(1,2,0).detach().cpu().numpy())) for i in range(images.shape[0])]

                aims_table.add_column("aims", data=log_images(images_aims))
                aims_table.add_column("enhanced", log_images(enhanced_images_aims))
                aims_table.add_column("enhancement", log_images(enhancement_aims))
                aims_table.add_column("D(enhanced) aims=1", D_out_d.mean(dim=(1,2,3)).detach().cpu().numpy())
                aims_table.add_column("cots", labels_aims.detach().cpu().numpy())

                kaggle_table = wandb.Table(columns=[], data=[])

                kaggle_table.add_column("kaggle", data=log_images(images_kaggle))
                kaggle_table.add_column("enhanced", log_images(enhanced_images_kaggle))
                kaggle_table.add_column("D(enhanced) aims=1", D_out_d_kaggle.mean(dim=(1,2,3)).detach().cpu().numpy())
                kaggle_table.add_column("cots", labels_kaggle.detach().cpu().numpy())
                # kaggle_table.add_column("P(cots)", pred_labels.sum(1).detach().cpu().numpy())

                wandb.log({'aims_table': aims_table, 'kaggle_table': kaggle_table})

            j += 1

            wandb.log({
                'iter': i_iter,
                'lr': optimizer.param_groups[0]['lr'],
                # 'num_steps': args.num_steps,
                'loss/lightnet': loss_lightnet_log,
                'loss/enhance': loss_enhance,
                'loss/discriminatior': loss_D_log,
                # 'loss/bounds': loss_enhance_bounds,
                'loss/enhance_ci': loss_enhance_ci,
                # 'loss/classifier': loss_classifier.item(),
                'mean_light': mean_light.item()
            })

            if not os.path.exists(f"imgs/{wandb.run.name}/"):
                os.makedirs(f"imgs/{wandb.run.name}/")

            if j % 50 == 0:
                torchvision.utils.save_image(images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims.png")
                torchvision.utils.save_image(images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle.png")
                torchvision.utils.save_image(enhanced_images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_enhanced.png")
                torchvision.utils.save_image(enhanced_images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_enhanced.png")
                torchvision.utils.save_image(enhancement_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_enhancement.png")


        scheduler.step()
        scheduler_D.step()

        # if i_iter % args.save_pred_every == 0 and i_iter != 0:
        print('taking snapshot ...')
        # torch.save(model.state_dict(), os.path.join(args.snapshot_dir, 'cots_classifier' + '.pth'))
        torch.save(lightnet.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_light_' + "latest" + '.pth'))
        torch.save(model_D.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_d_' + "latest" + '.pth'))
        # torch.save(model_D2.state_dict(), os.path.join(args.snapshot_dir, 'dannet_d2_' + str(i_iter) + '.pth'))

        if not os.path.exists(f"imgs/{wandb.run.name}/"):
            os.makedirs(f"imgs/{wandb.run.name}/")

        torchvision.utils.save_image(images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_{i_iter}.png")
        torchvision.utils.save_image(images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_{i_iter}.png")
        torchvision.utils.save_image(enhanced_images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_enhanced_{i_iter}.png")
        torchvision.utils.save_image(enhanced_images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_enhanced_{i_iter}.png")
        torchvision.utils.save_image(enhancement_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_enhancement_{i_iter}.png")

if __name__ == '__main__':
    main()
