import torch
import torch.nn as nn

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):from torch import nn, einsum
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
#             ]))
#     def forward(self, feats):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x

from mmdet.apis import inference_detector, init_detector
from mmyolo.utils import register_all_modules

register_all_modules()

class PatchPosEmbedding(nn.Module):
    def __init__(self, num_patches, dim) -> None:
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

    def forward(self, x):
        _, n, _ = x.shape
        # print("x shape", x.shape, "pos embedding", self.pos_embedding[:, :n].shape)
        x += self.pos_embedding[:, :n]

        return x

class YoloTransformerAdapter(nn.Module):
    def __init__(self, dim=256) -> None:
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

        del model

        patch_height, patch_width = 4, 4
        # used to assert feature sizes will be the same
        # self.orig_width, self.orig_height = 512, 512
        self.orig_width, self.orig_height = 1280, 768
        # image_sizes = [(256, 64, 64), (512, 32, 32), (768, 16, 16), (1024, 8, 8)]
        # image_sizes = [(256, 32, 32), (512, 16, 16), (768, 8, 8), (1024, 4, 4)]
        image_sizes = [(256, 96, 160), (512, 48, 80), (768, 24, 40), (1024, 12, 20)]

        self.dim = dim
        cross_heads = 4
        dim_cross_heads = 64
        dim_head = 64
        depth = 2
        heads = 4
        dropout = 0.01
        mlp_dim = dim
        num_latents = 256

        self.layers_per_size = nn.ModuleList([])
        for channels, height, width  in image_sizes:
            print("w,h", width, height)
            num_patches = (width // patch_width) * (height // patch_height)
            patch_dim = channels * patch_height * patch_width

            print("num patches", num_patches, "patch dim", patch_dim)

            to_patch_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
                # nn.Linear(patch_dim, dim),
                PatchPosEmbedding(num_patches, patch_dim)
            )

            self.layers_per_size.append(nn.ModuleList([
                to_patch_embedding,
                PreNorm(dim, Attention(dim, patch_dim, heads = cross_heads, dim_head = dim_cross_heads)),
                PreNorm(dim, FeedForward(dim, dim))
            ]))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))

        self.patches_out = nn.ModuleList([])
        self.queries_out = nn.ParameterList([])
        for channels, width, height in image_sizes:
            num_patches = (width // patch_width) * (height // patch_height)
            num_h = height // patch_height
            num_w = width // patch_width
            patch_dim = channels * patch_height * patch_width

            print("patches out", width, height, patch_width, patch_height, num_patches)

            print("patches out", f"b ({height * width}) ({patch_height * patch_width * channels}) -> b {channels} ({num_h * patch_height}) ({num_w * patch_width})")

            to_patch_embedding = nn.Sequential(
                nn.Linear(dim, patch_dim),
                Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_height, p2 = patch_width, h=num_h, w=num_w)
            )

            queries_dim = dim

            queries = nn.parameter.Parameter(torch.randn((num_patches, queries_dim)))
            decoder_cross_attn = PreNorm(queries_dim, Attention(queries_dim, context_dim=dim, heads = cross_heads, dim_head = 64), context_dim = dim)

            self.queries_out.append(queries)

            self.patches_out.append(nn.ModuleList([
                decoder_cross_attn,
                to_patch_embedding,
            ]))

        self.latents = nn.Parameter(torch.randn(num_latents, dim))

    def forward(self, x):
        assert list(x.shape[-2:]) == [self.orig_height, self.orig_width], f"{x.shape[-2:]} is not desired shape {[self.orig_height, self.orig_width]}"

        with torch.no_grad():
            feats: tuple = self.backbone(x)

        b = x.shape[0]

        # print('x shape', [x.shape for x in feats])
        # exit()

        # 0 torch.Size([1, 256, 64, 64])
        # 1 torch.Size([1, 512, 32, 32])
        # 2 torch.Size([1, 768, 16, 16])
        # 3 torch.Size([1, 1024, 8, 8])
        x = repeat(self.latents, 'n d -> b n d', b = b)

        for i, (embedding, cross_attn, cross_ff) in enumerate(self.layers_per_size):
            emb_feat = embedding(feats[i])
            x = cross_attn(x, context=emb_feat) + x
            x = cross_ff(x) + x
        
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        outs = []
        t_outs = []
        for i, (decoder_cross_atnn, to_patch_embedding) in enumerate(self.patches_out):
            queries = self.queries_out[i].unsqueeze(0).repeat((b, 1, 1))
            out = decoder_cross_atnn(queries, context=x)
            out = to_patch_embedding(out) # transformer learns residuals
            out = out.reshape(feats[i].shape)
            t_outs.append(out)
            out = out + feats[i]
            # print("feats[i]", feats[i].shape, out.shape)
            outs.append(out)

        return tuple(outs), x, t_outs

class FPNDiscriminator(nn.Module):
    def __init__(self, ndf=64, patched=True) -> None:
        super().__init__()

        self.patched = patched
        input_dim = sum([256, 512, 768, 1024])
        spec_size = (64, 64)

        self.upsample = nn.Upsample(size=spec_size)

        self.conv1 = nn.Conv2d(input_dim, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf*4, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*4, ndf*2, kernel_size=4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(ndf*2, ndf*2, kernel_size=4, stride=1, padding=1)
        if not self.patched:
            self.downsample = nn.Sequential(
                    nn.Conv2d(ndf*2, ndf*2, kernel_size=4, stride=1, padding=1),
                    nn.AdaptiveAvgPool2d((1, 1))
            )

            self.fc = nn.Sequential(
                    # nn.Linear(ndf*4, ndf),
                    # nn.ReLU(),
                    nn.Linear(ndf*2, 1)
            )
        else:
            self.classifier = nn.Conv2d(ndf*2, 1, kernel_size=4, stride=1, padding=1)

        self.soft = nn.Sigmoid() # disjoint events [dataset, real/fake]

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)



    def forward(self, x):
        fs = []
        for i, f in enumerate(x):
            fs.append(self.upsample(x[i]))

        # concat upsampled feature maps
        x = torch.cat(fs, dim=1) 

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