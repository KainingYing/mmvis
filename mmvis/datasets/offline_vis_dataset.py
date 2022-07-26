# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import tempfile
import zipfile
from collections import defaultdict
import random

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmtrack.core import eval_vis, results2outs
from mmdet.utils import get_root_logger

from .builder import DATASETS
from .base_vis_dataset import BaseVISDataset
from .coco_video_dataset import CocoVideoDataset
from mmtrack.core import eval_mot
from mmtrack.utils import get_root_logger
from .parsers import CocoVID
from .builder import DATASETS


@DATASETS.register_module(force=True)
class OfflineVISDataset(BaseVISDataset):
    """ Offline YouTube VIS dataset for video instance segmentation."""

    def __init__(self,
                 dataset_version,
                 load_as_video=True,
                 img_sampler=dict(
                     key_img_internal=1,
                     imgs_per_clip=2,
                     frame_range=20,
                     filter_key_img=True,
                     method='uniform'),
                 test_load_ann=False,
                 *args,
                 **kwargs):
        self.set_dataset_classes(dataset_version)
        self.load_as_video = load_as_video
        self.img_sampler = img_sampler
        self.test_load_ann = test_load_ann
        super(CocoVideoDataset, self).__init__(*args, **kwargs)
        self.logger = get_root_logger()

    def prepare_data(self, idx):
        """Get data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Data and annotations after pipeline with new keys introduced
            by pipeline.
        """
        img_info = self.data_infos[idx]  # select a key image
        img_infos = self.img_sampling(img_info, **self.img_sampler)  # sampling related frames in same video

        results = [self.prepare_results(img_info) for img_info in img_infos]
        return self.pipeline(results)

    def load_video_anns(self, ann_file):
        """Load annotations from COCOVID style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCOVID api.
        """
        self.coco = CocoVID(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        data_infos = []
        self.vid_ids = self.coco.get_vid_ids()
        self.img_ids = []
        for vid_id in self.vid_ids:
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            if self.img_sampler.get('key_img_internal', None):
                img_ids = img_ids[::self.img_sampler['key_img_internal']]
            self.img_ids.extend(img_ids)
            for img_id in img_ids:
                info = self.coco.load_imgs([img_id])[0]
                info['filename'] = info['file_name']
                data_infos.append(info)
        return data_infos

    def img_sampling(self,
                     img_info,
                     frame_range,
                     stride=1,
                     imgs_per_clip=1,
                     filter_key_img=True,
                     method='uniform',
                     return_key_img=True,
                     **kwargs):
        """Sampling frames in the same video for key frame."""
        # todo: simplify this process

        imgs_per_clip -= 1

        assert isinstance(img_info, dict)
        if isinstance(frame_range, int):
            assert frame_range >= 0, 'frame_range can not be a negative value.'
            frame_range = [-frame_range, frame_range]
        elif isinstance(frame_range, list):
            assert len(frame_range) == 2, 'The length must be 2.'
            assert frame_range[0] <= 0 and frame_range[1] >= 0
            for i in frame_range:
                assert isinstance(i, int), 'Each element must be int.'
        else:
            raise TypeError('The type of frame_range must be int or list.')

        if 'test' in method and \
                (frame_range[1] - frame_range[0]) != imgs_per_clip:
            print_log(
                'Warning:'
                "frame_range[1] - frame_range[0] isn't equal to num_ref_imgs."
                'Set num_ref_imgs to frame_range[1] - frame_range[0].',
                logger=self.logger)
            self.ref_img_sampler[
                'imgs_per_clip'] = frame_range[1] - frame_range[0]

        if (not self.load_as_video) or img_info.get('frame_id', -1) < 0 \
                or (frame_range[0] == 0 and frame_range[1] == 0):
            ref_img_infos = []
            for i in range(imgs_per_clip):
                ref_img_infos.append(img_info.copy())
        else:
            vid_id, img_id, frame_id = img_info['video_id'], img_info[
                'id'], img_info['frame_id']
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            left = max(0, frame_id + frame_range[0])
            right = min(frame_id + frame_range[1], len(img_ids) - 1)

            ref_img_ids = []
            if method == 'uniform':
                valid_ids = img_ids[left:right + 1]
                if filter_key_img and img_id in valid_ids:
                    valid_ids.remove(img_id)
                num_samples = min(imgs_per_clip, len(valid_ids))
                ref_img_ids.extend(random.sample(valid_ids, num_samples))
            elif method == 'bilateral_uniform':
                assert imgs_per_clip % 2 == 0, \
                    'only support load even number of ref_imgs.'
                for mode in ['left', 'right']:
                    if mode == 'left':
                        valid_ids = img_ids[left:frame_id + 1]
                    else:
                        valid_ids = img_ids[frame_id:right + 1]
                    if filter_key_img and img_id in valid_ids:
                        valid_ids.remove(img_id)
                    num_samples = min(imgs_per_clip // 2, len(valid_ids))
                    sampled_inds = random.sample(valid_ids, num_samples)
                    ref_img_ids.extend(sampled_inds)
            elif method == 'test_with_adaptive_stride':
                if frame_id == 0:
                    stride = float(len(img_ids) - 1) / (imgs_per_clip - 1)
                    for i in range(imgs_per_clip):
                        ref_id = round(i * stride)
                        ref_img_ids.append(img_ids[ref_id])
            elif method == 'test_with_fix_stride':
                if frame_id == 0:
                    for i in range(frame_range[0], 1):
                        ref_img_ids.append(img_ids[0])
                    for i in range(1, frame_range[1] + 1):
                        ref_id = min(round(i * stride), len(img_ids) - 1)
                        ref_img_ids.append(img_ids[ref_id])
                elif frame_id % stride == 0:
                    ref_id = min(
                        round(frame_id + frame_range[1] * stride),
                        len(img_ids) - 1)
                    ref_img_ids.append(img_ids[ref_id])
                img_info['num_left_ref_imgs'] = abs(frame_range[0]) \
                    if isinstance(frame_range, list) else frame_range
                img_info['frame_stride'] = stride
            else:
                raise NotImplementedError

            ref_img_infos = []
            for ref_img_id in ref_img_ids:
                ref_img_info = self.coco.load_imgs([ref_img_id])[0]
                ref_img_info['filename'] = ref_img_info['file_name']
                ref_img_infos.append(ref_img_info)
            # ref_img_infos = sorted(ref_img_infos, key=lambda i: i['frame_id'])

        return sorted([img_info, *ref_img_infos], key=lambda i: i['frame_id'])
