from mmdet.apis import inference_detector, init_detector
from mmyolo.utils import register_all_modules
import torch.nn as nn
from torchvision.models import resnet18
import torch
from torch.nn import init

class FCDiscriminator(nn.Module):
    def __init__(self, num_classes, ndf=64, patched=True):
        super(FCDiscriminator, self).__init__()

        self.patched = patched

        self.conv1 = nn.Conv2d(
            num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            ndf*2, ndf*4, kernel_size=4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(
            ndf*4, ndf*4, kernel_size=4, stride=1, padding=1)
        if not self.patched:
            self.downsample = nn.Sequential(
                nn.Conv2d(ndf*4, ndf*4, kernel_size=4, stride=1, padding=1),
                nn.AdaptiveAvgPool2d((1, 1))
            )

            self.fc = nn.Sequential(
                # nn.Linear(ndf*4, ndf),
                # nn.ReLU(),
                nn.Linear(ndf*4, 1)
            )
        else:
            self.classifier = nn.Conv2d(
                ndf*4, 1, kernel_size=4, stride=1, padding=1)

        self.soft = nn.Sigmoid()  # disjoint events [dataset, real/fake]

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)

        if not self.patched:
            x = self.downsample(x)
            x = x.flatten(1)
            x = self.fc(x)
        else:
            x = self.classifier(x)
        return self.soft(x)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

register_all_modules()


class YOLODiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        config = "yang_model/yolov5_l_kaggle_cots.py"
        ckpt = "yang_model/bbox_mAP_epoch_70.pth"

        model = init_detector(
            config=config,
            checkpoint=ckpt,
            device='cuda:0',
        )

        self.backbone: nn.Module = model.backbone
        self.backbone.eval()

        self.classifier = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=4, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def train(self, mode):
        super().train(mode)
        self.backbone.eval()

    def forward(self, x):

        with torch.no_grad():
            s4 = self.backbone(x)[3]

        return self.classifier(s4)


class ResNet18Discriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.resnet = resnet18(num_classes=2)

    def forward(self, x):
        o = self.resnet(x)
        print("resnet out", o.shape)
        return o
