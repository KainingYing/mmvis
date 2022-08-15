"""这个模仿的是SeqFormer中的旋转生成 pseudo 视频."""
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
    dict(type='SeqResize',
         share_params=True,
         multiscale_mode='value',
         img_scale=[(360, 1333), (480, 1333)],
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
coco2seq_train_pipeline = [
    dict(type='LoadMultiImagesFromFile', to_float32=True),
    dict(type='SeqLoadAnnotations',
         with_bbox=True,
         with_mask=True,
         with_track=True),
    dict(type='COCO2Seq', angle=30, num_frames=5, img_fill_val=0.),
    dict(type='SeqResize',
         share_params=True,
         multiscale_mode='value',
         override=True,
         img_scale=[(360, 1333), (480, 1333)],
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
data_root = 'data/youtube_vis_2021/'
dataset_version = '2021'

dict_coco = dict(
    type='OfflineVISDataset',
    dataset_version=dataset_version,
    ann_file=data_root + 'annotations/joint_coco_2021.json',
    img_prefix=data_root + 'train2017/',
    pipeline=coco2seq_train_pipeline,
    img_sampler=dict(key_img_internal=1,
                     imgs_per_clip=2,
                     frame_range=20,
                     filter_key_img=True,
                     method='uniform'),
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(type='ConcatDataset',
               separate_eval=True,
               datasets=[
                   dict(type='OfflineVISDataset',
                        dataset_version='2021',
                        ann_file='data/coco/' +
                        'annotations/joint_coco_2021.json',
                        img_prefix='data/coco/' + 'train2017/',
                        pipeline=coco2seq_train_pipeline,
                        img_sampler=dict(key_img_internal=1,
                                         imgs_per_clip=2,
                                         frame_range=20,
                                         filter_key_img=True,
                                         method='uniform')),
                   dict(type=dataset_type,
                        dataset_version=dataset_version,
                        ann_file=data_root +
                        'annotations/youtube_vis_2021_train.json',
                        img_prefix=data_root + 'train/JPEGImages',
                        img_sampler=dict(key_img_internal=1,
                                         imgs_per_clip=5,
                                         frame_range=20,
                                         filter_key_img=True,
                                         method='uniform'),
                        pipeline=train_pipeline)
               ]),
    val=dict(type=dataset_type,
             dataset_version=dataset_version,
             ann_file=data_root + 'annotations/youtube_vis_2021_valid.json',
             img_prefix=data_root + 'valid/JPEGImages',
             pipeline=test_pipeline),
    test=dict(type=dataset_type,
              dataset_version=dataset_version,
              ann_file=data_root + 'annotations/youtube_vis_2021_valid.json',
              img_prefix=data_root + 'valid/JPEGImages',
              pipeline=test_pipeline))
