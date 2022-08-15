# Copyright (c) OpenMMLab. All rights reserved.
import random
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import mmcv
import numpy as np
from mmcv.utils import print_log
from mmtrack.core.evaluation import eval_vis

from mmvis.utils import get_root_logger

from .base_vis_dataset import BaseVISDataset
from .builder import DATASETS
from .coco_video_dataset import CocoVideoDataset
from .parsers import CocoVID


@DATASETS.register_module(force=True)
class OfflineVISDataset(BaseVISDataset):
    """Offline YouTube VIS dataset for video instance segmentation."""
    def __init__(
            self,
            dataset_version,
            load_as_video=True,
            img_sampler=dict(key_img_internal=1,
                             imgs_per_clip=2,
                             frame_range=20,
                             filter_key_img=True,
                             method='uniform'),
            test_load_ann=False,
            # pipeline=None,
            *args,
            **kwargs):
        self.set_dataset_classes(dataset_version)
        self.load_as_video = load_as_video
        self.img_sampler = img_sampler
        self.test_load_ann = test_load_ann
        super(CocoVideoDataset, self).__init__(*args, **kwargs)
        self.logger = get_root_logger()
        # self.pipeline = Compose(pipeline)

    def prepare_data(self, idx):
        """Get data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Data and annotations after pipeline with new keys introduced
            by pipeline.
        """
        img_info = self.data_infos[idx]  # select a key image
        img_infos = self.img_sampling(
            img_info,
            **self.img_sampler)  # sampling related frames in same video

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
        self.videos = defaultdict(list)
        for vid_id in self.vid_ids:
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            if self.img_sampler.get('key_img_internal', None):
                img_ids = img_ids[::self.img_sampler['key_img_internal']]
            self.img_ids.extend(img_ids)
            for img_id in img_ids:
                info = self.coco.load_imgs([img_id])[0]
                info['filename'] = info['file_name']
                self.videos[info['video_id']].append(info)
                data_infos.append(info)
        self.videos = list(self.videos.values())

        # # 保存一个video的list
        # print('Create video list')
        # todo: 这个地方索引有一点慢，需要修改
        # self.videos = []
        # for video in self.coco.videos.values():
        #     frame_list = []
        #     idx = video['id']
        #     for data_info in data_infos:
        #         if idx == data_info['video_id']:
        #             frame_list.append(data_info)
        #     self.videos.append(frame_list)

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
                    ref_id = min(round(frame_id + frame_range[1] * stride),
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

    def prepare_test_img(self, idx):
        """Offline 方法直接输出一个视频."""
        video = self.videos[idx]
        results = [self.prepare_results(img_info) for img_info in video]
        results = self.pipeline(results)
        # 这里是hard code，只是为了兼容
        new_results = {}
        for key in results:
            new_results[key] = [results[key]]

        return new_results

    def __len__(self):
        if self.test_mode:
            return len(self.videos)
        else:
            return len(self.data_infos)

    def evaluate(self, results, metric=['segm'], logger=None):
        """Evaluation in COCO protocol.

        Args:
            results (dict): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'track_segm'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """
        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError('metric must be a list or a str.')
        allowed_metrics = ['segm']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')

        eval_results = dict()
        # save as zip file
        test_results = self.format_results(results, save_as_json=True)
        vis_results = self.convert_back_to_vis_format()
        track_segm_results = eval_vis(test_results, vis_results, logger)
        eval_results.update(track_segm_results)

        return eval_results

    def results2outs(self,
                     bbox_results=None,
                     mask_results=None,
                     mask_shape=None,
                     **kwargs):
        outputs = dict()

        if bbox_results is not None:
            labels = []
            for i, bbox in enumerate(bbox_results):
                labels.extend([i] * bbox.shape[0])
            labels = np.array(labels, dtype=np.int64)
            outputs['labels'] = labels

            bboxes = np.concatenate(bbox_results, axis=0).astype(np.float32)
            if bboxes.shape[1] == 5:
                outputs['bboxes'] = bboxes
                outputs['scores'] = bboxes[..., -1]
            elif bboxes.shape[1] == 6:
                ids = bboxes[:, 0].astype(np.int64)
                bboxes = bboxes[:, 1:]
                outputs['bboxes'] = bboxes
                outputs['ids'] = ids
            else:
                raise NotImplementedError(
                    f'Not supported bbox shape: (N, {bboxes.shape[1]})')

        if mask_results is not None:
            assert mask_shape is not None
            mask_height, mask_width = mask_shape
            mask_results = mmcv.concat_list(mask_results)
            if len(mask_results) == 0:
                masks = np.zeros((0, mask_height, mask_width)).astype(bool)
            else:
                masks = np.stack(mask_results, axis=0)
            outputs['masks'] = masks

        return outputs

    def format_results(self,
                       results,
                       resfile_path=None,
                       metrics=['segm'],
                       save_as_json=True):

        # if isinstance(metrics, str):
        #     metrics = [metrics]
        from mmvis.utils import ExpMetaInfo
        if ExpMetaInfo.get('work_dir', None):
            results_dir_path = Path(ExpMetaInfo['work_dir']) / 'results'
            mmcv.mkdir_or_exist(results_dir_path)
            resfile_path = results_dir_path
            Path(ExpMetaInfo['exp_name']).stem

        else:
            mmcv.mkdir_or_exist('outputs')
            resfile_path = Path('outputs')
        resfiles = resfile_path / 'results.json'

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        zip_file_name = resfile_path / f'{Path(ExpMetaInfo["exp_name"]).stem}_{timestamp}.zip'

        # inds = [i for i, _ in enumerate(self.data_infos) if _['frame_id'] == 0]
        # num_vids = len(inds)
        # assert num_vids == len(self.vid_ids)
        # inds.append(len(self.data_infos))
        vid_infos = self.coco.load_vids(self.vid_ids)

        json_results = []
        for i, video_pred in enumerate(results):
            bbox_pred, mask_pred = video_pred['ins_results']
            outs = self.results2outs(bbox_pred)
            masks = mmcv.concat_list(mask_pred)
            labels = outs['labels']
            scores = outs['scores']
            for label, score, segm in zip(labels, scores, masks):
                for ss in segm:
                    ss['counts'] = ss['counts'].decode()
                json_results.append({
                    'video_id': vid_infos[i]['id'],
                    'category_id': label.item() + 1,
                    'score': score.item(),
                    'segmentations': segm
                })

        if not save_as_json:
            return json_results
        mmcv.dump(json_results, resfiles)

        # zip the json file in order to submit to the test server.
        # zip_file_name = osp.join(resfile_path, 'submission_file.zip')

        with zipfile.ZipFile(zip_file_name,
                             mode='w',
                             compression=zipfile.ZIP_DEFLATED) as archive:
            print_log(f"\nzip the 'results.json' into '{zip_file_name}', "
                      'please submmit the zip file to the evaluation server.')
            archive.write(
                resfiles,
                arcname='results.json')  # arcname 用于将results.json直接压缩到根目录下面

        return json_results
