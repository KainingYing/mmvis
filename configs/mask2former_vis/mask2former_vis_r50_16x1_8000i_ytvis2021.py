_base_ = [
    '../_base_/datasets/offline_youtube_vis_2021.py',
    '../_base_/models/mask2former_vis_r50.py'
    '../_base_/default_runtime.py'
]

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# optimizer
optimizer = dict(type='AdamW',
                 lr=0.000025,
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
                         embed_multi},
                     norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[22000],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,  # no warmup
    warmup_iters=40)

# auto_scale_lr = dict(enable=True, base_batch_size=16)

max_iters = 32000
runner = dict(type='IterBasedRunner', max_iters=max_iters)

log_config = dict(interval=50,
                  hooks=[
                      dict(type='TextLoggerHook', by_epoch=False),
                      dict(type='TensorboardLoggerHook', by_epoch=False)])
interval = 32000
workflow = [('train', interval)]
checkpoint_config = dict(by_epoch=False,
                         interval=interval,
                         save_last=True,
                         max_keep_ckpts=3)

evaluation = dict(metric=['segm'], interval=interval)
