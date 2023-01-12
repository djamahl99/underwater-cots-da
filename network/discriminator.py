import torch.nn as nn
from torchvision.models import resnet18
import torch

class FCDiscriminator(nn.Module):
        def __init__(self, num_classes, ndf = 64, patched=True):
                super(FCDiscriminator, self).__init__()

                self.patched = patched

                self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
                self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
                self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=1, padding=1)
                self.conv4 = nn.Conv2d(ndf*4, ndf*4, kernel_size=4, stride=1, padding=1)
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
                        self.classifier = nn.Conv2d(ndf*4, 1, kernel_size=4, stride=1, padding=1)

                self.soft = nn.Sigmoid() # disjoint events [dataset, real/fake]

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

from mmdet.apis import inference_detector, init_detector
from mmyolo.utils import register_all_modules

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
