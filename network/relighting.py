import torch
import torch.nn as nn
import functools

affine_par = True


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ResnetGeneratorHalf(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGeneratorHalf, self).__init__()
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        l = 0

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
            l = int(ngf * mult / 2)
            break # ignore last convtranspose2d
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(l, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        o = self.model(input)

        return nn.functional.upsample(o, input.shape[-2:])


class GammaLightNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.unet = ResnetGenerator(3, 1, 64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=3)

    def forward(self, x):
        o = self.unet(x)
        # rgb = o[:, :3, :, :]
        gamma = o[:, 0, :, :].unsqueeze(1)


        gamma_correction = torch.pow(x, gamma)

        return gamma_correction - x

from torchvision.models import resnet34, ResNet34_Weights
class ResNetColorCorrector(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        resnet_out_dim = 1000

        ff_dim = 256

        self.contrast = nn.Sequential(
            nn.Linear(resnet_out_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, 3),
            nn.Sigmoid()
        )

        self.brightness = nn.Sequential(
            nn.Linear(resnet_out_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, 3)
        )

    def forward(self, x):
        image = x
        b = image.shape[0]

        x = self.resnet(x)
        con = self.contrast(x)
        br = self.brightness(x)

        con = con.reshape(b, 3, 1, 1)
        br = br.reshape(b, 3, 1, 1)

        con = torch.clamp(con, 0.01)
        adj_image = (image + br) * con

        residual = adj_image - image 

        return residual

import torchvision
class ResNetColorCorrector2(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        # self.resnet = nn.Sequential(
        #     resnet.conv1,
        #     resnet.bn1,
        #     resnet.relu,
        #     resnet.maxpool,
        #     #
        #     resnet.layer1,
        #     resnet.layer2,
        #     resnet.layer3,
        #     resnet.layer4
        #     # 
        # )

        self.resnet = ResnetGenerator(3, 32, 32, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=3)

        # resnet.layer4.

        l4_out_dim = 32

        # ff_dim = 256

        self.contrast = nn.Sequential(
            # nn.Conv2d(l4_out_dim, l4_out_dim, 7, stride=1),
            # nn.ReLU(),
            # nn.Conv2d(l4_out_dim, l4_out_dim, 3, stride=1),
            # nn.ReLU(),
            nn.Conv2d(l4_out_dim, 3, 3, stride=1),
            nn.Sigmoid()
        )

        self.brightness = nn.Sequential(
            # nn.Conv2d(l4_out_dim, l4_out_dim, 7, stride=1),
            # nn.ReLU(),
            # nn.Conv2d(l4_out_dim, l4_out_dim, 3, stride=1),
            # nn.ReLU(),
            nn.Conv2d(l4_out_dim, 3, 3, stride=1),
        )

    def forward(self, x):
        image = x
        b = image.shape[0]

        x = self.resnet(x)
        con = self.contrast(x)
        br = self.brightness(x)
        con = nn.functional.upsample(con, size=tuple(image.shape[-2:]))
        br = nn.functional.upsample(br, size=tuple(image.shape[-2:]))

        con = torch.clamp(con, 0.01)
        # con *= 2.0
        adj_image = image * con + br

        # residual = adj_image - image 

        return adj_image

def LightNet(ngf=64, n_blocks=3):
    # model = ResnetGeneratorHalf(3, 3, 64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=3)
    model = ResnetGenerator(3, 3, ngf, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=n_blocks)
    return model

class LightNetTransparent(nn.Module):
    def __init__(self, ngf=64, n_blocks=3) -> None:
        super().__init__()

        self.n_blocks = n_blocks
        self.ngf = ngf

        self.generator = ResnetGenerator(3, 4, ngf=ngf, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=n_blocks)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.generator(x)

        x_rgb = x[:, :3, :, :]
        x_a = x[:, -1, :, :].unsqueeze(1)

        x_a = self.sigmoid(x_a)
    
        # x_a = x_a.repeat((1, 3, 1, 1))
        return x_rgb, x_a

# def LightNetTransparent(ngf=64, n_blocks=3):
#     model = ResnetGenerator(3, 4, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=n_blocks)



#     return model

class L_grayscale(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.loss = nn.L1Loss()

        
        self.normx = nn.BatchNorm2d(3, track_running_stats=False, affine=False)
        self.normt = nn.BatchNorm2d(3, track_running_stats=False, affine=False)

    def forward(self, x, t):
        x = self.normx(x)
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        L_x = 0.299*r + 0.587*g + 0.114*b

        t = self.normt(t)
        r, g, b = t[:, 0], t[:, 1], t[:, 2]
        L_t = 0.299*r + 0.587*g + 0.114*b
        
        return self.loss(L_x, L_t)

from .ciconv2d import CIConv2d
class L_ColorInvarianceConv(nn.Module):
    def __init__(self, invariant) -> None:
        super().__init__()

        self.invariant = invariant
        self.ci_conv = CIConv2d(invariant)
        self.ci_conv.eval()

        self.loss = nn.L1Loss()

    def forward(self, x, t):
        x = self.ci_conv(x)
        t = self.ci_conv(t)

        return self.loss(x, t)

class L_exp_z(nn.Module):
    def __init__(self, patch_size):
        super(L_exp_z, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, x, mean_val):
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean - torch.FloatTensor([mean_val]).cuda(), 2))
        return d

class Loss_bounds(nn.Module):
    def __init__(self) -> None:
        super(Loss_bounds, self).__init__()

        self.relu = nn.GELU()

    def forward(self, x):
        v =  torch.mean(self.relu(-x) + self.relu(x-1))
        return v # scale


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)