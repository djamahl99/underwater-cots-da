# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections import OrderedDict

import torch

convert_dict_p5 = {
    'model.0': 'backbone.stem',
    'model.1': 'backbone.stage1.0',
    'model.2': 'backbone.stage1.1',
    'model.3': 'backbone.stage2.0',
    'model.4': 'backbone.stage2.1',
    'model.5': 'backbone.stage3.0',
    'model.6': 'backbone.stage3.1',
    'model.7': 'backbone.stage4.0',
    'model.8': 'backbone.stage4.1',
    'model.9.cv1': 'backbone.stage4.2.conv1',
    'model.9.cv2': 'backbone.stage4.2.conv2',
    'model.10': 'neck.reduce_layers.2',
    'model.13': 'neck.top_down_layers.0.0',
    'model.14': 'neck.top_down_layers.0.1',
    'model.17': 'neck.top_down_layers.1',
    'model.18': 'neck.downsample_layers.0',
    'model.20': 'neck.bottom_up_layers.0',
    'model.21': 'neck.downsample_layers.1',
    'model.23': 'neck.bottom_up_layers.1',
    'model.24.m': 'bbox_head.head_module.convs_pred',
}

convert_dict_p5_reverse = {convert_dict_p5[k] for k in convert_dict_p5.keys()}

convert_dict_p6 = {
    'model.0': 'backbone.stem',
    'model.1': 'backbone.stage1.0',
    'model.2': 'backbone.stage1.1',
    'model.3': 'backbone.stage2.0',
    'model.4': 'backbone.stage2.1',
    'model.5': 'backbone.stage3.0',
    'model.6': 'backbone.stage3.1',
    'model.7': 'backbone.stage4.0',
    'model.8': 'backbone.stage4.1',
    'model.9': 'backbone.stage5.0',
    'model.10': 'backbone.stage5.1',
    'model.11.cv1': 'backbone.stage5.2.conv1',
    'model.11.cv2': 'backbone.stage5.2.conv2',
    'model.12': 'neck.reduce_layers.3',
    'model.15': 'neck.top_down_layers.0.0',
    'model.16': 'neck.top_down_layers.0.1',
    'model.19': 'neck.top_down_layers.1.0',
    'model.20': 'neck.top_down_layers.1.1',
    'model.23': 'neck.top_down_layers.2',
    'model.24': 'neck.downsample_layers.0',
    'model.26': 'neck.bottom_up_layers.0',
    'model.27': 'neck.downsample_layers.1',
    'model.29': 'neck.bottom_up_layers.1',
    'model.30': 'neck.downsample_layers.2',
    'model.32': 'neck.bottom_up_layers.2',
    'model.33.m': 'bbox_head.head_module.convs_pred',
}

convert_dict_p6_reverse = {convert_dict_p6[k]: k for k in convert_dict_p6.keys()}


def convert(src, dst):
    """Convert keys in pretrained YOLOv5 models to mmyolo style."""
    # if src.endswith('6.pt'):
    convert_dict = convert_dict_p6_reverse
    is_p6_model = True
    print('Converting P6 model')
    # else:
    #     convert_dict = convert_dict_p5
    #     is_p6_model = False
    #     print('Converting P5 model')
    try:
        model = torch.load(src)
        # blobs = yolov5_model.state_dict()
        print(model.keys())
        blobs = model['state_dict']

        print(blobs.keys())
    except ModuleNotFoundError:
        raise RuntimeError(
            'This script must be placed under the ultralytics/yolov5 repo,'
            ' because loading the official pretrained model need'
            ' `model.py` to build model.')
    state_dict = OrderedDict()

    for key, weight in blobs.items():
        print(key)
        num, module = key.split('.')[1:3]
        print("num / module", num, module)
        # if (is_p6_model and
        #     (num == '11' or num == '33')) or (not is_p6_model and
        #                                       (num == '9' or num == '24')):
        #     if module == 'anchors':
        #         continue
        #     prefix = f'model.{num}.{module}'
        # else:
        #     prefix = f'model.{num}'

        # new_key = key.replace(prefix, convert_dict[prefix])
        new_key = key
        for k in convert_dict.keys():
            new_key = new_key.replace(k, convert_dict[k])

        if '.blocks.' in new_key:
            new_key = new_key.replace('.blocks.', '.m.')
            new_key = new_key.replace('.conv', '.cv')
            new_key = new_key.replace('.cv.', '.conv.') # revert w/o numbers
        else:
            new_key = new_key.replace('.main_conv', '.cv1', )
            new_key = new_key.replace('.short_conv', '.cv2')
            new_key = new_key.replace('.final_conv', '.cv3')

        # new_key = "model.model." + new_key

        print("old key", key)
        print("new_key", new_key)

        state_dict[new_key] = weight
        print(f'Convert {key} to {new_key}')

    # save checkpoint
    checkpoint = dict()
    checkpoint['state_dict'] = state_dict

    # model = torch.hub.load('ultralytics/yolov5', 'yolov5-p6')
    import sys
    sys.path.append("../yolov5")
    from models.yolo import DetectionModel


    model = DetectionModel(cfg='../yolov5/models/hub/yolov5-p6.yaml', nc=1)

    nsd = model.state_dict()

    # copy anchors from WrappedYolo
    from network.yolo_wrapper import WrappedYOLO
    m = WrappedYOLO()
    anchors = torch.tensor(m.bbox_head.prior_generator.base_sizes)
    priors_base_sizes = torch.tensor(
            m.bbox_head.prior_generator.base_sizes, dtype=torch.float)
    featmap_strides = torch.tensor(
        m.bbox_head.featmap_strides, dtype=torch.float)[:, None, None]

    anchors = priors_base_sizes / featmap_strides
    # anchors = torch.tensor([[(19, 27), (44, 40), (38, 94)],
    #                     [(96, 68), (86, 152), (180, 137)],
    #                     [(140, 301), (303, 264), (238, 542)],
    #                     [(436, 615), (739, 380), (925, 792)]])
    # print([(k, nsd[k]) for k in nsd.keys() if '33' in k])
    print("example anchors", nsd['model.33.anchors'])
    print("anchors", anchors)

    # copy the anchors over
    # state_dict['model.33.anchors'] = nsd['model.33.anchors']
    state_dict['model.33.anchors'] = anchors

    for k in nsd.keys():
        if k not in state_dict:
            print([k_ for k_ in state_dict.keys() if k[:10] in k_])

            raise Exception(f"{k} not in state_dict")


    model.load_state_dict(state_dict)
    checkpoint['model'] = model

    torch.save(checkpoint, dst)


# Note: This script must be placed under the yolov5 repo to run.
def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument(
        '--src', default='mmyolo.pth', help='src yolov5 model path')
    parser.add_argument('--dst', default='yang_yolov5.pt', help='save path')
    args = parser.parse_args()
    convert(args.src, args.dst)


if __name__ == '__main__':
    main()
