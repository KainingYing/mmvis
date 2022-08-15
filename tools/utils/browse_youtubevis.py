# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from collections.abc import Sequence
from pathlib import Path

import mmcv
from mmcv import Config, DictAction
from mmdet.core.utils import mask2ndarray
from mmdet.utils import update_data_root

from mmvis.core.utils import imshow_tracks
from mmvis.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Browse youtube-vis dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--skip-type',
                        type=str,
                        nargs='+',
                        default=[
                            'SeqDefaultFormatBundle', 'VideoCollect',
                            'SeqNormalize', 'OfflineSeqDefaultFormatBundle',
                            'SeqRename'
                        ],
                        help='skip some useless pipeline')
    parser.add_argument('--save-type',
                        default='folder',
                        choices=['mp4', 'folder'],
                        help='')
    parser.add_argument(
        '--output-dir',
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type, cfg_options):
    def skip_pipeline_steps(config):
        config['pipeline'] = [
            x for x in config.pipeline if x['type'] not in skip_type
        ]

    cfg = Config.fromfile(config_path)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg and train_data_cfg[
            'type'] != 'MultiImageMixDataset':
        train_data_cfg = train_data_cfg['dataset']

    if isinstance(train_data_cfg, Sequence):
        [skip_pipeline_steps(c) for c in train_data_cfg]
    else:
        skip_pipeline_steps(train_data_cfg)

    return cfg


def main():
    args = parse_args()
    # cfg = Config.fromfile(args.config)
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)
    # 采样整个视频片段，需要将imgs_per_clip设置得很长
    cfg.data.train.img_sampler.imgs_per_clip = 999

    # FOLDER_OUT = (args.save_type == 'folder')

    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset.coco.videos))
    # progress_bar = mmcv.ProgressBar(len(dataset))

    for _, video in dataset.coco.vidToImgs.items():
        img_list = dataset[video[0]['id'] - 1]
        # video_name = img_list[0]['ori_filename'].split('/')[0]

        for img_id, img in enumerate(img_list):
            gt_bboxes = img['gt_bboxes']
            gt_labels = img['gt_labels']
            gt_instance_ids = img['gt_instance_ids']

            gt_masks = img.get('gt_masks', None)

            # 这一步是为了coco2seq做准备，因为coco2seq中的图片都是一样的名字
            out_file = Path(args.output_dir) / (img['ori_filename'][:-4] +
                                                f'_{img_id}.jpg')

            if gt_masks is not None:
                gt_masks = mask2ndarray(gt_masks)

            imshow_tracks(img['img'],
                          gt_bboxes,
                          gt_labels,
                          gt_instance_ids,
                          gt_masks,
                          backend='plt',
                          classes=dataset.CLASSES,
                          out_file=out_file,
                          show=False)

        progress_bar.update()


if __name__ == '__main__':
    main()
