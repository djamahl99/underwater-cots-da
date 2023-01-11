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

from dataset.zurich_pair_dataset import zurich_pair_DataSet
from dataset.cityscapes_dataset import cityscapesDataSet
from configs.train_config import get_arguments

import wandb

import torchvision
from network.discriminator import ResNet18Discriminator, YOLODiscriminator

from network.relighting import GammaLightNet, L_ColorInvarianceConv, L_grayscale, Loss_bounds
from network.transformer import YoloTransformerAdapter, FPNDiscriminator

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
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda")

    cudnn.enabled = True
    cudnn.benchmark = True

    transformer = YoloTransformerAdapter()
    transformer = nn.DataParallel(transformer)
    transformer = transformer.to(device)
    transformer.train()

    model_D = FPNDiscriminator()
    model_D = nn.DataParallel(model_D)
    model_D.train()
    model_D.to(device)

    model_D_cots = FPNDiscriminator(patched=False)
    model_D_cots = nn.DataParallel(model_D_cots)
    model_D_cots.train()
    model_D_cots.to(device)

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    ds = kaggle_aims_pair()
    trainloader = data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    trainloader_iter = enumerate(trainloader)

    optimizer = optim.Adam(list(transformer.parameters()) + list(model_D_cots.parameters()), lr=args.learning_rate, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    optimizer.zero_grad()

    optimizer_D = optim.Adam(model_D.parameters(), lr=args.learning_rate_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    loss_bce = nn.BCELoss()

    kaggle_label = 0.0
    aims_label = 1.0
    fake_label = 0.0
    real_label = 1.0

    #r_coeff = 1.0

    wandb.init(
        project="DANNet (transformer)",
        config=dict(
            transformer_dim=transformer.dim,
            discriminator=model_D.module._get_name(),
            cots_classifier=model_D_cots.module._get_name(),
        )
    )

    wandb.watch(models=[transformer, model_D])

    for i_iter in range(args.num_steps):
        optimizer.zero_grad()
        optimizer_D.zero_grad()

        j = 0
        for images_kaggle, images_aims, labels_kaggle, labels_aims in tqdm(trainloader, desc="Batch"):
            labels_kaggle = labels_kaggle.to(device)

            if len(labels_kaggle.shape) == 1:
                labels_kaggle.reshape(images_kaggle.shape[0], -1)

            images_aims = images_aims.to(device)
            mean_light = images_kaggle.mean()
            
            images_kaggle = images_kaggle.to(device)

            print("kaggle imgs", images_kaggle.shape,"aims", images_aims.shape)

            b_size = images_kaggle.shape[0]

            ########################################################################################################
            # Train DISCRIMINATOR 
            ########################################################################################################
            for p in transformer.parameters():
                p.requires_grad_(False)
            for p in model_D.parameters():
                p.requires_grad_(True)
            
            model_D.zero_grad()

            aims_feats, latent_dim_aims = transformer(images_aims)
            kaggle_feats, latent_dim_kaggle = transformer(images_kaggle)

            # D with aims
            D_out_d = model_D(aims_feats)
            D_label_d = torch.FloatTensor(D_out_d.data.size()).fill_(aims_label).to(device) # use kaggle as want lightnet to produce kaggle style
            loss_adv_aims = loss_bce(D_out_d, D_label_d)

            # D with kaggle
            D_out_d = model_D(kaggle_feats)
            D_label_d = torch.FloatTensor(D_out_d.data.size()).fill_(kaggle_label).to(device) # use kaggle as want lightnet to produce kaggle style
            loss_adv_kaggle = loss_bce(D_out_d, D_label_d)

            loss = loss_adv_aims + loss_adv_kaggle

            loss_D_log = loss.item()
            
            loss.backward()
            optimizer_D.step()

            ########################################################################################################
            # Train Transformer
            ########################################################################################################
            for p in transformer.parameters():
                p.requires_grad_(True)
            for p in model_D.parameters():
                p.requires_grad_(False)

            transformer.zero_grad()

            aims_feats, latent_dim_aims = transformer(images_aims)
            kaggle_feats, latent_dim_kaggle = transformer(images_kaggle)

            # D with aims
            D_out_d = model_D(aims_feats)
            D_label_d = torch.FloatTensor(D_out_d.data.size()).fill_(kaggle_label).to(device) # use kaggle as want lightnet to produce kaggle style
            loss_adv_aims = loss_bce(D_out_d, D_label_d)

            # D with kaggle
            D_out_d = model_D(kaggle_feats)
            D_label_d = torch.FloatTensor(D_out_d.data.size()).fill_(kaggle_label).to(device) # use kaggle as want lightnet to produce kaggle style
            loss_adv_kaggle = loss_bce(D_out_d, D_label_d)

            loss_transformer = loss_adv_aims + loss_adv_kaggle

            ########################################################################################################
            # Train COTS classifier (should change this to training yolo model)
            ########################################################################################################

            preds_cls = model_D_cots(kaggle_feats)
            print("labels kaggle", labels_kaggle.shape, preds_cls.shape)
            loss_preds_kaggle = loss_bce(preds_cls, labels_kaggle)

            loss_classifier = loss_preds_kaggle

            loss = loss_transformer + loss_classifier
            
            loss.backward()
            optimizer.step()

            j += 1

            wandb.log({
                'iter': i_iter,
                # 'num_steps': args.num_steps,
                'loss/classifier': loss_classifier.item(),
                'loss/transformer': loss_transformer.item(),
                'loss/discriminator': loss_D_log
            })

        # if i_iter % args.save_pred_every == 0 and i_iter != 0:
        print('taking snapshot ...')
        torch.save(transformer.state_dict(), os.path.join(args.snapshot_dir, 'transformer' + '.pth'))
        torch.save(model_D_cots.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_d_cots_' + "latest" + '.pth'))
        torch.save(model_D.state_dict(), os.path.join(args.snapshot_dir, f'{wandb.run.name}_d_' + "latest" + '.pth'))


if __name__ == '__main__':
    main()
