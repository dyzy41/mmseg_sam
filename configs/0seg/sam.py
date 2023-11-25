_base_ = [
    '../_base_/datasets/whub_sam.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

metainfo = dict(
                classes=('background', 'building'),
                palette=[[0, 0, 0], [255, 255, 255]])

data_root = '/home/ps/HDD/zhaoyq_data/SegDATA/whub_seg_1024'
crop_size = (1024, 1024)

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
model = dict(
    type='BuildSAM',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='NullDecodeHead',
        in_channels=256,
        channels=512,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # auxiliary_head=dict(
    #     type='FCNHead',
    #     in_channels=256,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))


train_dataloader = dict(
    batch_size=1,
    num_workers=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.txt'))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='val.txt'))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test.txt'))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# training schedule for 20k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=20000, save_best='mIoU'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))