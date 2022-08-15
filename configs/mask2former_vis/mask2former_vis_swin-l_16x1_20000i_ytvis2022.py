_base_ = [
    '../_base_/datasets/offline_youtube_vis_2022.py',
    '../_base_/default_runtime.py'
]
num_classes = 40
model = dict(
    type='Mask2FormerVIS',
    backbone=dict(type='SwinTransformer',
                  pretrain_img_size=384,
                  embed_dims=192,
                  depths=[2, 2, 18, 2],
                  num_heads=[6, 12, 24, 48],
                  window_size=12,
                  mlp_ratio=4,
                  qkv_bias=True,
                  qk_scale=None,
                  drop_rate=0.,
                  attn_drop_rate=0.,
                  drop_path_rate=0.3,
                  patch_norm=True,
                  out_indices=(0, 1, 2, 3),
                  with_cp=False,
                  convert_weights=True,
                  frozen_stages=-1,
                  init_cfg=None),
    head=dict(type='Mask2FormerVISHead',
              in_channels=[192, 384, 768, 1536],
              strides=[4, 8, 16, 32],
              feat_channels=256,
              out_channels=256,
              num_classes=num_classes,
              num_queries=200,
              num_transformer_feat_level=3,
              pixel_decoder=dict(
                  type='MSDeformAttnPixelDecoder',
                  num_outs=3,
                  norm_cfg=dict(type='GN', num_groups=32),
                  act_cfg=dict(type='ReLU'),
                  encoder=dict(type='DetrTransformerEncoder',
                               num_layers=6,
                               transformerlayers=dict(
                                   type='BaseTransformerLayer',
                                   attn_cfgs=dict(
                                       type='MultiScaleDeformableAttention',
                                       embed_dims=256,
                                       num_heads=8,
                                       num_levels=3,
                                       num_points=4,
                                       im2col_step=128,
                                       dropout=0.0,
                                       batch_first=False,
                                       norm_cfg=None,
                                       init_cfg=None),
                                   ffn_cfgs=dict(type='FFN',
                                                 embed_dims=256,
                                                 feedforward_channels=1024,
                                                 num_fcs=2,
                                                 ffn_drop=0.0,
                                                 act_cfg=dict(type='ReLU',
                                                              inplace=True)),
                                   operation_order=('self_attn', 'norm', 'ffn',
                                                    'norm')),
                               init_cfg=None),
                  positional_encoding=dict(type='SinePositionalEncoding',
                                           num_feats=128,
                                           normalize=True),
                  init_cfg=None),
              enforce_decoder_input_project=False,
              positional_encoding=dict(type='SinePositionalEncoding3D',
                                       num_feats=128,
                                       normalize=True),
              transformer_decoder=dict(
                  type='DetrTransformerDecoder',
                  return_intermediate=True,
                  num_layers=9,
                  transformerlayers=dict(
                      type='DetrTransformerDecoderLayer',
                      attn_cfgs=dict(type='MultiheadAttention',
                                     embed_dims=256,
                                     num_heads=8,
                                     attn_drop=0.0,
                                     proj_drop=0.0,
                                     dropout_layer=None,
                                     batch_first=False),
                      ffn_cfgs=dict(embed_dims=256,
                                    feedforward_channels=2048,
                                    num_fcs=2,
                                    act_cfg=dict(type='ReLU', inplace=True),
                                    ffn_drop=0.0,
                                    dropout_layer=None,
                                    add_identity=True),
                      feedforward_channels=2048,
                      operation_order=('cross_attn', 'norm', 'self_attn',
                                       'norm', 'ffn', 'norm')),
                  init_cfg=None),
              loss_cls=dict(type='CrossEntropyLoss',
                            use_sigmoid=False,
                            loss_weight=2.0,
                            reduction='mean',
                            class_weight=[1.0] * num_classes + [0.1]),
              loss_mask=dict(type='CrossEntropyLoss',
                             use_sigmoid=True,
                             reduction='mean',
                             loss_weight=5.0),
              loss_dice=dict(type='DiceLoss',
                             use_sigmoid=True,
                             activate=True,
                             reduction='mean',
                             naive_dice=True,
                             eps=1.0,
                             loss_weight=5.0)),
    train_cfg=dict(num_points=12544,
                   oversample_ratio=3.0,
                   importance_sample_ratio=0.75,
                   assigner=dict(type='MaskHungarianAssigner',
                                 cls_cost=dict(type='ClassificationCost',
                                               weight=2.0),
                                 mask_cost=dict(type='CrossEntropyLossCost',
                                                weight=5.0,
                                                use_sigmoid=True),
                                 dice_cost=dict(type='DiceCost',
                                                weight=5.0,
                                                pred_act=True,
                                                eps=1.0)),
                   sampler=dict(type='MaskPseudoSampler')))

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# optimizer
optimizer = dict(type='AdamW',
                 lr=0.0001,
                 weight_decay=0.05,
                 eps=1e-8,
                 betas=(0.9, 0.999),
                 paramwise_cfg=dict(custom_keys={
                     'backbone':
                     dict(lr_mult=0.1, decay_mult=1.0),
                     'query_embed':
                     embed_multi,
                     'query_feat':
                     embed_multi,
                     'level_embed':
                     embed_multi,
                 },
                                    norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[5500, 10000],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10)

max_iters = 20000
runner = dict(type='IterBasedRunner', max_iters=max_iters)

log_config = dict(interval=50,
                  hooks=[
                      dict(type='TextLoggerHook', by_epoch=False),
                      dict(type='TensorboardLoggerHook', by_epoch=False)
                  ])
interval = 8000
workflow = [('train', interval)]
checkpoint_config = dict(by_epoch=False,
                         interval=interval,
                         save_last=True,
                         max_keep_ckpts=3)

evaluation = dict(metric=['segm'], interval=interval)

load_from = 'checkpoints/mask2former_swin-l-p4-w12-384-in21k_lsj_16x1_100e_coco-panoptic_20220407_104949-d4919c44.pth'
