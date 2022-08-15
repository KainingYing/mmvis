# dataset settings
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True)
train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations',
         with_bbox=True,
         with_mask=True,
         with_track=True),
    dict(
        type='SeqResize',
        share_params=True,
        multiscale_mode='value',
        img_scale=[(360, 1333), (480, 1333)],
        # img_scale=[(360, 1333), (480, 1333), (512, 1333), (544, 1333), (576, 1333),
        #            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
        #            (736, 1333), (768, 1333), (800, 1333)],
        keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='VideoCollect',
         keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks',
               'gt_instance_ids']),
    dict(type='OfflineSeqDefaultFormatBundle'),
    dict(type='SeqRename', mapping=dict(img='video', img_metas='video_metas'))
]
test_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqResize',
         share_params=True,
         keep_ratio=True,
         multiscale_mode='value',
         img_scale=(360, 1333)),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=32),
    dict(type='VideoCollect', keys=['img']),
    dict(type='OfflineMultiImagesToTensor'),
    dict(type='SeqRename', mapping=dict(img='video', img_metas='video_metas'))
]
dataset_type = 'OfflineVISDataset'
data_root = 'data/youtube_vis_2019/'
dataset_version = data_root[-5:-1]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(type=dataset_type,
               dataset_version=dataset_version,
               ann_file=data_root + 'annotations/youtube_vis_2019_train.json',
               img_prefix=data_root + 'train/JPEGImages',
               img_sampler=dict(key_img_internal=1,
                                imgs_per_clip=2,
                                frame_range=20,
                                filter_key_img=True,
                                method='uniform'),
               pipeline=train_pipeline),
    val=dict(type=dataset_type,
             dataset_version=dataset_version,
             ann_file=data_root + 'annotations/youtube_vis_2019_valid.json',
             img_prefix=data_root + 'valid/JPEGImages',
             pipeline=test_pipeline),
    test=dict(type=dataset_type,
              dataset_version=dataset_version,
              ann_file=data_root + 'annotations/youtube_vis_2019_valid.json',
              img_prefix=data_root + 'valid/JPEGImages',
              pipeline=test_pipeline))
