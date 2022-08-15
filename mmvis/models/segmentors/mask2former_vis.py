# Copyright (c) ZJUTCV. All rights reserved.
from mmcv.runner import auto_fp16
from mmdet.core import bbox2result
from mmdet.models import BaseDetector

from ..builder import SEGMENTORS, build_backbone, build_head, build_neck


@SEGMENTORS.register_module(force=True)
class Mask2FormerVIS(BaseDetector):
    """Mask2Former for Video Instance Segmentation.

    Refer to https://github.com/facebookresearch/Mask2Former.
    """
    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(Mask2FormerVIS, self).__init__(init_cfg=init_cfg)

        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        head_ = head.copy()
        head_.update(train_cfg=train_cfg)
        head_.update(test_cfg=test_cfg)

        self.head = build_head(head_)

        self.num_classes = self.head.num_classes

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def forward_dummy(self, video):
        batch_size, num_frames, _, height, width = video.shape
        dummy_img_metas = [[
            dict(batch_input_shape=(height, width),
                 img_shape=(height, width, 3)) for _ in range(num_frames)
        ] for _ in range(batch_size)]

        imgs = video.flatten(0, 1)  # resize video into (bs*T) x 3 x H x W
        x = self.extract_feat(imgs)
        outs = self.head(x, dummy_img_metas)
        return outs

    def aug_test(self, imgs, img_metas, **kwargs):
        pass

    def simple_test(self, video, video_metas, **kwargs):
        imgs = video.flatten(0, 1)  # resize video into (bs*T) x 3 x H x W
        x = self.extract_feat(imgs)

        results = self.head.simple_test(x, video_metas, **kwargs)

        for i in range(len(results)):

            if 'ins_results' in results[i]:
                labels_per_image, bboxes, mask_pred_binary = results[i][
                    'ins_results']
                bbox_results = bbox2result(bboxes, labels_per_image,
                                           self.num_classes)
                mask_results = [[] for _ in range(self.num_classes)]
                for j, label in enumerate(labels_per_image):
                    mask = mask_pred_binary[j].detach().cpu().numpy()
                    mask_results[label].append(mask)
                results[i]['ins_results'] = bbox_results, mask_results

        return results

    @auto_fp16(apply_to=('video', ))
    def forward(self, video, video_metas, return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(video, video_metas, **kwargs)
        else:
            return self.forward_test(video, video_metas, **kwargs)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""

        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, video, video_metas, gt_bboxes, gt_labels, gt_masks,
                      gt_instance_ids, **kargs):
        # add batch_input_shape in img_metas
        batch_input_shape = tuple(video[0].size()[-2:])
        for img_metas in video_metas:
            for img_meta in img_metas:
                img_meta['batch_input_shape'] = batch_input_shape

        imgs = video.flatten(0, 1)  # resize video into (bs*T) x 3 x H x W
        x = self.extract_feat(imgs)
        losses = self.head.forward_train(x, video_metas, gt_bboxes, gt_labels,
                                         gt_masks, gt_instance_ids)

        return losses

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(loss=loss,
                       log_vars=log_vars,
                       num_samples=len(data['video_metas']))

        return outputs

    def forward_test(self, videos, video_metas, **kwargs):

        for var, name in [(videos, 'videos'), (video_metas, 'video_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(videos)
        if num_augs != len(video_metas):
            raise ValueError(f'num of augmentations ({len(videos)}) '
                             f'!= num of video meta ({len(video_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for video, video_meta in zip(videos, video_metas):
            pass
            # batch_size = len(img_meta)
            batch_input_shape = tuple(video.size()[-2:])
            for img_metas in video_meta:
                for img_meta in img_metas:
                    img_meta['batch_input_shape'] = batch_input_shape

            # batch_size = len(img_meta)
            # for img_id in range(batch_size):
            #     img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(videos[0], video_metas[0], **kwargs)
        else:
            assert videos[0].size(0) == 1, 'aug test does not support ' \
                                           'inference with batch size ' \
                                           f'{videos[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(videos, video_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass
