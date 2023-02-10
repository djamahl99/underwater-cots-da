import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils import data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from dataset.kaggle_aims_pair_dataset import kaggle_aims_pair
from dataset.kaggle_aims_pair_dataset_w_boxes import kaggle_aims_pair_boxed
import numpy as np
from network import *

from configs.train_config import get_arguments

import wandb

import torchvision
from network.Modules_chutak import GlobalGenerator
from network.discriminator import ResNet18Discriminator, YOLODiscriminator

from network.relighting import L_ColorInvarianceConv, L_grayscale, LightNetWSementation, StyleLossStep

from network.deeprelight_networks import MultiscaleDiscriminator, define_G, VGGLoss

from network.yolo_wrapper import WrappedYOLO

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

def print_img_stats(img):

    print("min max", img.min(), img.max())
    img = np.clip(img, a_min=0, a_max=1)
    return img

class StatDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.pool(x)
        x = x.reshape(-1, 3)

        return self.classifier(x)

def main():
    args = get_arguments()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda")

    cudnn.enabled = True
    cudnn.benchmark = True


    # lightnet = LightNet(ngf=64)
    lightnet = LightNetWSementation()
    # lightnet = GlobalGenerator(input_nc=3, output_nc=3)
    # init_weights(lightnet)
    # lightnet = nn.DataParallel(lightnet)
    lightnet.eval()
    lightnet.to(device)

    # darknet = LightNet(ngf=64)
    darknet = LightNetWSementation()
    # darknet = GlobalGenerator(input_nc=3, output_nc=3)
    # init_weights(darknet)
    # darknet = nn.DataParallel(darknet)
    darknet.eval()
    darknet.to(device)

    model_D = NLayerDiscriminator(input_nc=3, ndf=64)
    # model_D = MultiscaleDiscriminator(input_nc=3, ndf=64)
    # model_D = FCDiscriminator(3)
    # model_D = nn.DataParallel(model_D)
    
    model_D.train()
    model_D.to(device)

    model_D2 = StatDiscriminator()
    model_D2.train()
    model_D2.to(device)

    # eval on aims
    ds_val_aims = kaggle_aims_pair_boxed(aims_split="val.json")
    yolo = WrappedYOLO()
    yolo.eval()
    yolo.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    ds = kaggle_aims_pair()
    trainloader = data.DataLoader(
        ds, batch_size=4, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    trainloader_iter = enumerate(trainloader)

    args.learning_rate = 5e-4
    args.learning_rate_D = 1e-4
    args.num_steps = 1000
    optimizer_light = optim.Adam(list(lightnet.parameters()), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    optimizer_light.zero_grad()

    optimizer_dark = optim.Adam(list(darknet.parameters()), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    optimizer_dark.zero_grad()

    optimizer_D = optim.Adam(list(model_D.parameters()) + list(model_D2.parameters()), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    loss_bce = nn.BCEWithLogitsLoss()

    loss_style = StyleLossStep()

    kaggle_label = 0.0
    aims_label = 1.0
    fake_label = 0.0
    real_label = 1.0

    wandb.init(
        project="UNet (DANNet) adversarial AIMS -> Kaggle",
        config={
            'lightnet': lightnet,#.module,
            'darknet': darknet,#.module,
            'e_wgts': loss_style.e_wgts
        }
    )

    if not os.path.exists(f"imgs/{wandb.run.name}/"):
        os.makedirs(f"imgs/{wandb.run.name}/")

    wandb.watch(models=[lightnet])

    mean_light_ema = torch.tensor(0.5).to(device)
    ema_v = 0.99

    mean_dark_ema = torch.tensor(0.4).to(device)

    mean_lightnet_out = torch.tensor(0.5).to(device)
    mean_darknet_out = torch.tensor(0.5).to(device)



    for i_iter in range(args.num_steps):
        optimizer_dark.zero_grad()
        optimizer_light.zero_grad()
        optimizer_D.zero_grad()

        j = 0
        for images_kaggle, images_aims, labels_kaggle, labels_aims in tqdm(trainloader, desc="Batch"):

            images_aims = images_aims.to(device)
            with torch.no_grad():
                mean_light = images_kaggle.mean()
                mean_light_ema = mean_light_ema * ema_v + (1 - ema_v) * mean_light

                mean_dark = images_aims.mean()
                mean_dark_ema = mean_dark_ema * ema_v + (1 - ema_v) * mean_dark
            
            # to loss
            mean_light = mean_light_ema
            mean_dark = mean_dark_ema
            
            images_kaggle = images_kaggle.to(device)

            b_size = images_kaggle.shape[0]

            ########################################################################################################
            # DISCRIMINATOR 
            ########################################################################################################
            for p in lightnet.parameters():
                p.requires_grad_(False)
            for p in darknet.parameters():
                p.requires_grad_(False)
            for p in model_D.parameters():
                p.requires_grad_(True)
            for p in model_D2.parameters():
                p.requires_grad_(True)
            
            model_D.zero_grad()
            model_D2.zero_grad()
            model_D.train()
            model_D2.train()
            lightnet.eval()
            darknet.eval()
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

            # D with REAL aims
            D_out_d = model_D(images_aims)
            D_label_d = torch.FloatTensor(D_out_d.data.size()).fill_(aims_label).to(device)
            loss_adv_real_aims = loss_bce(D_out_d, D_label_d)

            D_out_d2 = model_D2(images_aims)
            D_label_d2 = torch.FloatTensor(D_out_d2.data.size()).fill_(aims_label).to(device)
            loss_adv_real_aims2 = loss_bce(D_out_d2, D_label_d2)

            # D with enhanced aims
            D_out_d = model_D(brighter_images_aims)
            loss_adv_brighter_aims = loss_bce(D_out_d, D_label_d)

            # D_out_d2 = model_D2(brighter_images_aims)
            # loss_adv_brighter_aims2 = loss_bce(D_out_d2, D_label_d2)

            # # D with cycle aims
            # D_out_d = model_D(cycle_images_aims)
            # loss_adv_cycle_aims = loss_bce(D_out_d, D_label_d)

            # D with REAL kaggle
            D_out_d = model_D(images_kaggle)
            D_label_d = torch.FloatTensor(D_out_d.data.size()).fill_(kaggle_label).to(device)
            loss_adv_real_kaggle = loss_bce(D_out_d, D_label_d)

            D_out_d2 = model_D2(images_kaggle)
            loss_adv_real_kaggle2 = loss_bce(D_out_d2, D_label_d2)

            # D with enhanced kaggle
            D_out_d = model_D(darker_images_kaggle)
            loss_adv_darker_kaggle = loss_bce(D_out_d, D_label_d)

            # D_out_d2 = model_D2(darker_images_kaggle)
            # loss_adv_darker_kaggle2 = loss_bce(D_out_d2, D_label_d2)

            # # D with cycle kaggle
            # D_out_d = model_D(cycle_images_kaggle)
            # loss_adv_cycle_kaggle = loss_bce(D_out_d, D_label_d)

            loss = loss_adv_real_aims + loss_adv_real_kaggle + loss_adv_brighter_aims + loss_adv_darker_kaggle #+ loss_adv_cycle_kaggle + loss_adv_cycle_aims
            loss_d2 = loss_adv_real_aims2 + loss_adv_real_kaggle2 #+ loss_adv_brighter_aims2 + loss_adv_darker_kaggle2

            loss_log_d1 = loss.item()
            loss_log_d2 = loss_d2.item()

            loss += loss_d2
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
            for p in model_D.parameters():
                p.requires_grad_(False)

            lightnet.train()
            darknet.eval()
            model_D.eval()
            lightnet.zero_grad()
            # model.zero_grad()

            # segmentation loss
            lightnet_seg_aims = lightnet.forward_seg(images_aims)
            D_label_d = torch.FloatTensor(lightnet_seg_aims.data.size()).fill_(1.0).to(device)
            
            loss_seg = loss_bce(lightnet_seg_aims, D_label_d)

            lightnet_seg_kaggle = lightnet.forward_seg(images_kaggle)
            D_label_d = torch.FloatTensor(lightnet_seg_kaggle.data.size()).fill_(0.0).to(device)
            loss_seg += loss_bce(lightnet_seg_kaggle, D_label_d)


            loss_seg_log_lightnet = loss_seg.item()

            # don't train seg using enhancement
            loss_seg.backward()
            optimizer_light.step()
            for p in lightnet.resnet.parameters():
                p.requires_grad_(False)

            loss = loss_style(lightnet, darknet, model_D, model_D2, mean_light, mean_dark, images_aims, images_kaggle, label=kaggle_label)

            loss_lightnet_log = loss.item()

            # loss += loss_seg * 10
            loss.backward()
            optimizer_light.step()

            ########################################################################################################
            # DARKNET
            ########################################################################################################
            for p in lightnet.parameters():
                p.requires_grad_(False)
            for p in darknet.parameters():
                p.requires_grad_(True)
            for p in model_D.parameters():
                p.requires_grad_(False)

            darknet.train()
            lightnet.eval()
            model_D.eval()
            darknet.zero_grad()


            # segmentation loss
            darknet_seg_aims = darknet.forward_seg(images_aims)
            D_label_d = torch.FloatTensor(darknet_seg_aims.data.size()).fill_(0.0).to(device)
            
            loss_seg = loss_bce(darknet_seg_aims, D_label_d)

            darknet_seg_kaggle = darknet.forward_seg(images_kaggle)
            D_label_d = torch.FloatTensor(darknet_seg_kaggle.data.size()).fill_(1.0).to(device)
            loss_seg += loss_bce(darknet_seg_kaggle, D_label_d)


            loss_seg_log_darknet = loss_seg.item()

            # don't train seg with enhancement
            loss_seg.backward()
            optimizer_dark.step()
            for p in darknet.resnet.parameters():
                p.requires_grad_(False)

            loss = loss_style(darknet, lightnet, model_D, model_D2, mean_dark, mean_light, images_kaggle, images_aims, label=aims_label)

            loss_darknet_log = loss.item()

            # loss += loss_seg * 10
            loss.backward()
            optimizer_dark.step()

            # predictions
            if (j == len(trainloader) - 1 or j % 500 == 0) and j > 0:
                aims_table = wandb.Table(columns=[], data=[])

                #print_img_stats = lambda image: print("min max", image.min(), image.max())

                # log_images = lambda images: [wandb.Image(print_img_stats(images[i].permute(1,2,0).detach().cpu().numpy())) for i in range(images.shape[0])]

                # aims_table.add_column("aims", data=log_images(images_aims))
                # aims_table.add_column("brightened", log_images(brighter_images_aims))
                # # aims_table.add_column("darkened", log_images(darker_images_))

                # kaggle_table = wandb.Table(columns=[], data=[])

                # kaggle_table.add_column("kaggle", data=log_images(images_kaggle))
                # # kaggle_table.add_column("brightened", log_images(brightened_images_kaggle))
                # kaggle_table.add_column("darkened", log_images(darker_images_kaggle))

                # wandb.log({'aims_table': aims_table, 'kaggle_table': kaggle_table})

                # evaluate
                map50 = evaluate(yolo, ds_val_aims, enhance=lightnet)
                wandb.log({"val/aims/map50": map50})

            j += 1

            wandb.log({
                'iter': i_iter,
                'lr/dark': optimizer_dark.param_groups[0]['lr'],
                'lr/light': optimizer_light.param_groups[0]['lr'],
                'lr/discriminator': optimizer_D.param_groups[0]['lr'],
                'loss/d1': loss_log_d1,
                'loss/d2': loss_log_d2,
                'loss/lightnet': loss_lightnet_log,
                'loss/darknet': loss_darknet_log,
                # 'loss/enhance': loss_enhance_bright + loss_enhance_dark,
                'loss/discriminator': loss_D_log,
                'loss/darknet_seg': loss_seg_log_darknet,
                'loss/lightnet_seg': loss_seg_log_lightnet,
                # 'loss/enhance_ci': loss_enhance_ci_dark + loss_enhance_ci_bright,
                'mean_light': mean_light.item(),
                'mean_dark': mean_dark.item(),
                'mean_lightnet': mean_lightnet_out.item(),
                'mean_darknet': mean_darknet_out.item()
            })

            if not os.path.exists(f"imgs/{wandb.run.name}/"):
                os.makedirs(f"imgs/{wandb.run.name}/")

            if j % 50 == 0:
                torchvision.utils.save_image(images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims.png")
                torchvision.utils.save_image(images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle.png")
                torchvision.utils.save_image(darker_images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_darker.png")
                torchvision.utils.save_image(lightnet_seg_aims.detach().cpu(), f"imgs/{wandb.run.name}/lightnet_seg_aims.png")
                torchvision.utils.save_image(darknet_seg_aims.detach().cpu(), f"imgs/{wandb.run.name}/darknet_seg_aims.png")
                torchvision.utils.save_image(lightnet_seg_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/lightnet_seg_kaggle.png")
                torchvision.utils.save_image(darknet_seg_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/darknet_seg_kaggle.png")
                # torchvision.utils.save_image(darkened_images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_darker.png")
                torchvision.utils.save_image(brighter_images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_brighter.png")
                # torchvision.utils.save_image(brightened_images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_brighter.png")


        #adjust_learning_rate(args, optimizer_dark, i_iter)
        #adjust_learning_rate(args, optimizer_light, i_iter)
        #adjust_learning_rate_D(args, optimizer_D, i_iter)

        print('taking snapshot ...')
        torch.save(lightnet.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_light_' + "latest" + '.pth'))
        torch.save(darknet.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_dark_' + "latest" + '.pth'))
        torch.save(model_D.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_d_' + "latest" + '.pth'))

        torchvision.utils.save_image(images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_{i_iter}.png")
        torchvision.utils.save_image(images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_{i_iter}.png")
        torchvision.utils.save_image(darker_images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_darker_{i_iter}.png")
        # torchvision.utils.save_image(darkened_images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_darker_{i_iter}.png")
        torchvision.utils.save_image(brighter_images_aims.detach().cpu(), f"imgs/{wandb.run.name}/aims_brighter_{i_iter}.png")
        #torchvision.utils.save_image(brightened_images_kaggle.detach().cpu(), f"imgs/{wandb.run.name}/kaggle_brighter_{i_iter}.png")

if __name__ == '__main__':
    main()
