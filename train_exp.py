import torch
from PIL import Image
from mmdet.apis import inference_detector, init_detector
from mmyolo.utils import register_all_modules
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from typing import Optional, List, Tuple, Dict
from torch import Tensor
from mmengine.structures import InstanceData


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

def weighted_sample_mean(values: torch.Tensor, weights: torch.Tensor, dim=1):
    weights = weights.reshape(-1, 1) / values.sum()
    print('values shape', values.shape, weights.shape)
    num = (weights * values).sum(axis=dim) # TODO: change sum dimension
    dnm = weights.sum()

    if dnm == 0.0:
        return 0.0 * values.sum(dim=dim)

    return num / dnm

class RandAugment(torch.nn.Module):
    r"""RandAugment data augmentation method based on
    `"RandAugment: Practical automated data augmentation with a reduced search space"
    <https://arxiv.org/abs/1909.13719>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_ops (int): Number of augmentation transformations to apply sequentially.
        magnitude (int): Magnitude for all the transformations.
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """

    def __init__(
        self,
        num_ops: int = 2,
        magnitude: int = 9,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.num_ops = num_ops
        self.magnitude = magnitude
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int, image_size: Tuple[int, int]) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.3, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.3, num_bins), True),
        #     "TranslateX": (torch.linspace(0.0, 150.0 / 331.0 * image_size[1], num_bins), True),
        #     "TranslateY": (torch.linspace(0.0, 150.0 / 331.0 * image_size[0], num_bins), True),
        #     "Rotate": (torch.linspace(0.0, 30.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Color": (torch.linspace(0.0, 0.9, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.9, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.9, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 4)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins, (height, width))
        for _ in range(self.num_ops):
            op_index = int(torch.randint(len(op_meta), (1,)).item())
            op_name = list(op_meta.keys())[op_index]
            magnitudes, signed = op_meta[op_name]
            magnitude = float(magnitudes[self.magnitude].item()) if magnitudes.ndim > 0 else 0.0
            if signed and torch.randint(2, (1,)):
                magnitude *= -1.0
            img = _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

        return img


    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_ops={self.num_ops}"
            f", magnitude={self.magnitude}"
            f", num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s

def main():
        register_all_modules()

        config = "yolov5_l_kaggle_cots.py"
        ckpt = "work_dirs/yolov5_l_kaggle_cots/bbox_mAP_epoch_70.pth"

        model = init_detector(
                config=config,
                checkpoint=ckpt,
                device='cuda',
        )

        device = torch.device("cuda")


        backbone = model.backbone.to(device)
        neck = model.neck.to(device)
        head = model.bbox_head.to(device)

        image = torch.zeros((1, 3, 768, 1280)).to(device)

        path = "/home/etc004/code/DANNet/AIMS_data_test/data/cots_CCIP/COTS_Trip_Data_Sep2022/20210907_033643_Seq04-Reef 21-551-Reef 21-551 - 2/20210907_041145_000_1704.jpg"
        image = Image.open(path)

        t = transforms.Compose([
                transforms.Resize((768, 1280)),
        ])

        t_tensor = transforms.ToTensor() 

        aug = transforms.RandAugment()
        num_augs = 1

        image = t(image)

        auged_imgs = [t_tensor(image).unsqueeze(0)]
        for i in range(num_augs):
                auged_imgs.append(t_tensor(aug(image)).unsqueeze(0))

        auged_imgs = torch.cat(auged_imgs, dim=0) 

        auged_imgs = (auged_imgs).to(device)

        print("auged_imgs shape", auged_imgs.shape)

        with torch.no_grad():
                x = backbone(auged_imgs)
                x = neck(x)
                x = head(x)

        img_metas = [dict(ori_shape=(768,1280), scale_factor=1, batch_input_shape=(768,1280)) for _ in range(auged_imgs.shape[0])]
        preds = head.predict_by_feat(x[0], x[1], x[2], img_metas,\
                rescale=False, with_nms=True)
        print("preds", preds)

        #### PLOTTING
        plt.figure(figsize=(16,9))
        #define Matplotlib figure and axis
        # fig, ax = plt.subplots()
        ax = plt.subplot(1,1,1)

        ax.imshow(auged_imgs[0].permute(1, 2, 0).detach().cpu().numpy())

        line_styles = ['-', '--', '-.', ':']

        box_num = 0


        bboxes = []
        scores = []

        for i in range(len(preds)):
                for j in range(len(preds[i]['bboxes'])):
                        bbox = preds[i]['bboxes'][j].detach().cpu().numpy()
                        
                        print("score", preds[i]['scores'][j])

                        bbox = xyxy2matplotlibxywh(bbox)

                        bboxes.append(bbox)
                        scores.append(preds[i]['scores'][j].detach().cpu().numpy())

                        print("box", bbox)

                        label = "cots"

                        #add rectangle to plot
                        ax.add_patch(Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], alpha=1, fill=False, lw=1, label=label, color=[c/255 for c in PALETTE[box_num]]))
                        box_num += 1

        # avg_box = np.array(bboxes).mean(0)

        avg_box = weighted_sample_mean(np.array(bboxes), np.array(scores), dim=0)
        print("avg box", avg_box.shape)
        print("var boxes", np.array(bboxes).var(axis=0))
        

        ax.add_patch(Rectangle((avg_box[0], avg_box[1]), avg_box[2], avg_box[3], alpha=0.25, fill=True, lw=4, label="average", color=[c/255 for c in PALETTE[-1]]))


        print("auged_imgs[0].unsqueeze(0)", auged_imgs[0].unsqueeze(0).shape)
        # try loss
        print("backbone:")
        # x = backbone(auged_imgs[0].unsqueeze(0))
        x = backbone(auged_imgs)
        print("then neck:")
        x = neck(x)
        print("then head:")
        x = head(x)
        print('x', len(x))
        print("then head loss:")
        # l = head.loss(x, batch_data_samples=dict(img_metas=[img_metas[0]], bbox_labels=[{'bboxes': [avg_box], 'labels':[0]}]))


        instance_data = InstanceData(img_metas=[img_metas[0]])
        instance_data.bboxes = torch.tensor([avg_box]).to(device)
        instance_data.labels = torch.tensor([0]).to(device)
        print('d test', instance_data.bboxes.shape, instance_data.labels.shape)

        # loss_inputs = x + ([dict(bboxes=[avg_box], labels=[0])], [img_metas[0]])
        loss_inputs = x + ([instance_data], [img_metas[0]])
        losses = head.loss_by_feat(*loss_inputs)
        print("losses", losses)

        plt.legend()
        plt.show()

if __name__ == "__main__":
        main()