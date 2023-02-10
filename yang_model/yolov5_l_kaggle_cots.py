default_scope = 'mmyolo'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=1),
    param_scheduler=dict(
        type='YOLOv5ParamSchedulerHook',
        scheduler_type='linear',
        lr_factor=0.1,
        max_epochs=300),
    checkpoint=dict(
        type='CheckpointHook', interval=10, save_best='auto',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='mmdet.DetVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet.DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = 'pretrained_models/yolov5_l-p6-v62_syncbn_fast_8xb16-300e_coco_20221027_234308-7a2ba6bf.pth'
resume = False
file_client_args = dict(backend='disk')
data_root = '/mnt/cruncher-ph/ssd/datasets-ml/Kaggle_1080_google_v1/'
dataset_type = 'YOLOv5CocoDataset'
num_classes = 1
img_scale = (1280, 1280)
deepen_factor = 1.0
widen_factor = 1.0
max_epochs = 300
save_epoch_intervals = 10
train_batch_size_per_gpu = 4
train_num_workers = 6
val_batch_size_per_gpu = 1
val_num_workers = 2
persistent_workers = True
batch_shapes_cfg = dict(
    type='BatchShapePolicy',
    batch_size=1,
    img_size=1280,
    size_divisor=64,
    extra_pad_ratio=0.5)
anchors = [[(19, 27), (44, 40), (38, 94)], [(96, 68), (86, 152), (180, 137)],
           [(140, 301), (303, 264), (238, 542)],
           [(436, 615), (739, 380), (925, 792)]]
strides = [8, 16, 32, 64]
num_det_layers = 4
model = dict(
    type='YOLODetector',
    data_preprocessor=dict(
        type='YOLOv5DetDataPreprocessor',
        mean=[0.0, 0.0, 0.0],
        std=[255.0, 255.0, 255.0],
        bgr_to_rgb=True),
    backbone=dict(
        type='YOLOv5CSPDarknet',
        deepen_factor=1.0,
        widen_factor=1.0,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
        arch='P6',
        out_indices=(2, 3, 4, 5)),
    neck=dict(
        type='YOLOv5PAFPN',
        deepen_factor=1.0,
        widen_factor=1.0,
        in_channels=[256, 512, 768, 1024],
        out_channels=[256, 512, 768, 1024],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='YOLOv5Head',
        head_module=dict(
            type='YOLOv5HeadModule',
            num_classes=1,
            in_channels=[256, 512, 768, 1024],
            widen_factor=1.0,
            featmap_strides=[8, 16, 32, 64],
            num_base_priors=3),
        prior_generator=dict(
            type='mmdet.YOLOAnchorGenerator',
            base_sizes=[[(19, 27), (44, 40), (38, 94)],
                        [(96, 68), (86, 152), (180, 137)],
                        [(140, 301), (303, 264), (238, 542)],
                        [(436, 615), (739, 380), (925, 792)]],
            strides=[8, 16, 32, 64]),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.22499999999999998),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-07,
            reduction='mean',
            loss_weight=0.037500000000000006,
            return_iou=True),
        loss_obj=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=2.0999999999999996),
        prior_match_thr=4.0,
        obj_level_weights=[4.0, 1.0, 0.25, 0.06]),
    test_cfg=dict(
        multi_label=True,
        nms_pre=30000,
        score_thr=0.1,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=300))
albu_train_transforms = [
    dict(type='Blur', p=0.2),
    dict(type='CLAHE', p=0.5),
    dict(type='Transpose', p=0.5),
    dict(type='RandomGamma', p=0.3),
    dict(type='MotionBlur', p=0.3),
    dict(type='GaussNoise', p=0.3),
    dict(type='HueSaturationValue', p=0.5),
    dict(type='ShiftScaleRotate', p=0.5),
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=0.5,
        contrast_limit=0.5,
        p=0.5)
]
pre_transform = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True)
]
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Mosaic',
        img_scale=(1280, 1280),
        pad_val=114.0,
        pre_transform=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        prob=1.0),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.5, 1.5),
        border=(-640, -640),
        border_val=(114, 114, 114)),
    dict(
        type='mmdet.Albu',
        transforms=[
            dict(type='Blur', p=0.2),
            dict(type='CLAHE', p=0.5),
            dict(type='Transpose', p=0.5),
            dict(type='RandomGamma', p=0.3),
            dict(type='MotionBlur', p=0.3),
            dict(type='GaussNoise', p=0.3),
            dict(type='HueSaturationValue', p=0.5),
            dict(type='ShiftScaleRotate', p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=0.5,
                contrast_limit=0.5,
                p=0.5)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap=dict(img='image', gt_bboxes='bboxes')),
    dict(type='YOLOv5HSVRandomAug'),
    dict(
        type='YOLOv5MixUp',
        img_scale=(1280, 1280),
        ratio_range=(0.3, 2),
        pad_val=114.0,
        pre_transform=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        prob=0.5),
    dict(
        type='mmdet.CutOut', cutout_shape=[(5, 5), (8, 8), (16, 16)],
        prob=0.5),
    dict(
        type='mmdet.RandomFlip',
        flip_ratio=[0.33, 0.33, 0.33],
        direction=['horizontal', 'vertical', 'diagonal'],
        prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=6,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='/mnt/cruncher-ph/ssd/datasets-ml/Kaggle_1080_google_v1/',
        ann_file='mmdet_split_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Mosaic',
                img_scale=(1280, 1280),
                pad_val=114.0,
                pre_transform=[
                    dict(
                        type='LoadImageFromFile',
                        file_client_args=dict(backend='disk')),
                    dict(type='LoadAnnotations', with_bbox=True)
                ]),
            dict(
                type='YOLOv5RandomAffine',
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(0.09999999999999998, 1.9),
                border=(-640, -640),
                border_val=(114, 114, 114)),
            dict(
                type='YOLOv5MixUp',
                prob=0.1,
                pre_transform=[
                    dict(
                        type='LoadImageFromFile',
                        file_client_args=dict(backend='disk')),
                    dict(type='LoadAnnotations', with_bbox=True),
                    dict(
                        type='Mosaic',
                        img_scale=(1280, 1280),
                        pad_val=114.0,
                        pre_transform=[
                            dict(
                                type='LoadImageFromFile',
                                file_client_args=dict(backend='disk')),
                            dict(type='LoadAnnotations', with_bbox=True)
                        ]),
                    dict(
                        type='YOLOv5RandomAffine',
                        max_rotate_degree=0.0,
                        max_shear_degree=0.0,
                        scaling_ratio_range=(0.09999999999999998, 1.9),
                        border=(-640, -640),
                        border_val=(114, 114, 114))
                ]),
            dict(
                type='mmdet.Albu',
                transforms=[
                    dict(type='Blur', p=0.01),
                    dict(type='MedianBlur', p=0.01),
                    dict(type='ToGray', p=0.01),
                    dict(type='CLAHE', p=0.01)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
                keymap=dict(img='image', gt_bboxes='bboxes')),
            dict(type='YOLOv5HSVRandomAug'),
            dict(type='mmdet.RandomFlip', prob=0.5),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'flip', 'flip_direction'))
        ],
        metainfo=dict(CLASSES=('COTS', ), PALETTE=[(220, 20, 60)])),
    collate_fn=dict(type='yolov5_collate'))
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='YOLOv5KeepRatioResize', scale=(1280, 1280)),
    dict(
        type='LetterResize',
        scale=(1280, 1280),
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='/mnt/cruncher-ph/ssd/datasets-ml/Kaggle_1080_google_v1/',
        test_mode=True,
        data_prefix=dict(img=''),
        ann_file='mmdet_split_test.json',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='YOLOv5KeepRatioResize', scale=(1280, 1280)),
            dict(
                type='LetterResize',
                scale=(1280, 1280),
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param'))
        ],
        batch_shapes_cfg=dict(
            type='BatchShapePolicy',
            batch_size=1,
            img_size=1280,
            size_divisor=64,
            extra_pad_ratio=0.5),
        metainfo=dict(CLASSES=('COTS', ), PALETTE=[(220, 20, 60)])))
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='YOLOv5CocoDataset',
        data_root='/mnt/cruncher-ph/ssd/datasets-ml/Kaggle_1080_google_v1/',
        test_mode=True,
        data_prefix=dict(img=''),
        ann_file='mmdet_split_test.json',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='YOLOv5KeepRatioResize', scale=(1280, 1280)),
            dict(
                type='LetterResize',
                scale=(1280, 1280),
                allow_scale_up=False,
                pad_val=dict(img=114)),
            dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor', 'pad_param'))
        ],
        batch_shapes_cfg=dict(
            type='BatchShapePolicy',
            batch_size=1,
            img_size=1280,
            size_divisor=64,
            extra_pad_ratio=0.5),
        metainfo=dict(CLASSES=('COTS', ), PALETTE=[(220, 20, 60)])))
param_scheduler = [
    dict(type='PolyLR', power=0.9, eta_min=0.0001, by_epoch=True)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.005,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=4),
    constructor='YOLOv5OptimizerConstructor',
    clip_grad=None)
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=
    '/mnt/cruncher-ph/ssd/datasets-ml/Kaggle_1080_google_v1/mmdet_split_test.json',
    metric='bbox')
test_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=
    '/mnt/cruncher-ph/ssd/datasets-ml/Kaggle_1080_google_v1/mmdet_split_test.json',
    metric='bbox')
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=300, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
lr_factor = 0.1
affine_scale = 0.9
mosaic_affine_pipeline = [
    dict(
        type='Mosaic',
        img_scale=(1280, 1280),
        pad_val=114.0,
        pre_transform=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True)
        ]),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.09999999999999998, 1.9),
        border=(-640, -640),
        border_val=(114, 114, 114))
]
val_img_scale = (1280, 1280)
metainfo = dict(CLASSES=('COTS', ), PALETTE=[(220, 20, 60)])
launcher = 'none'
work_dir = './work_dirs/yolov5_l_kaggle_cots'
