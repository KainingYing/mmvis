# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(
        type='SeqLoadAnnotations',
        with_bbox=True,
        with_mask=True,
        with_track=True),
    dict(
        type='SeqResize',
        share_params=True,
        multiscale_mode='value',
        img_scale=[(640, 360), (960, 480), (600, 1200), (800, 1333)],
        keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'gt_instance_ids']),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 360),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='VideoCollect', keys=['img'])
        ])
]
dataset_type = 'OfflineVISDataset'
data_root = 'data/youtube_vis_2021/'
dataset_version = data_root[-5:-1]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        dataset_version=dataset_version,
        ann_file=data_root + 'annotations/youtube_vis_2021_train_sub.json',
        img_prefix=data_root + 'train/JPEGImages',
        img_sampler=dict(
            key_img_internal=1,
            imgs_per_clip=999,
            frame_range=20,
            filter_key_img=True,
            method='uniform'),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        dataset_version=dataset_version,
        ann_file=data_root + 'annotations/youtube_vis_2021_valid.json',
        img_prefix=data_root + 'valid/JPEGImages',
        ref_img_sampler=None,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        dataset_version=dataset_version,
        ann_file=data_root + 'annotations/youtube_vis_2021_valid.json',
        img_prefix=data_root + 'valid/JPEGImages',
        ref_img_sampler=None,
        pipeline=test_pipeline))