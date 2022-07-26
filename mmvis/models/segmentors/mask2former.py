import mmcv
import numpy as np

from mmdet.core import INSTANCE_OFFSET, bbox2result
from mmdet.core.visualization import imshow_det_bboxes
from mmcv.runner import BaseModule

from ..builder import SEGMENTORS, build_backbone, build_head, build_neck


@SEGMENTORS.register_module(force=True)
class Mask2Former(BaseModule):
    """
    Mask2Former for Video Instance Segmentation

    Refer to https://github.com/facebookresearch/Mask2Former.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(Mask2Former, self).__init__(init_cfg=init_cfg)

        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        # head_ = head.copy()
        # head_.update(train_cfg=train_cfg)
        # head_.update(test_cfg=test_cfg)
        #
        # self.head = build_head(head_)
        #
        # self.num_classes = self.head.num_classes
        #
        # self.train_cfg = train_cfg
        # self.test_cfg = test_cfg

    def forward_dummy(self):
        pass

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      videos,
                      video_metas,
                      gt_video,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg=None,
                      gt_bboxes_ignore=None,
                      **kargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[Dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            gt_masks (list[BitmapMasks]): true segmentation masks for each box
                used if the architecture supports a segmentation task.
            gt_semantic_seg (list[tensor]): semantic segmentation mask for
                images for panoptic segmentation.
                Defaults to None for instance segmentation.
            gt_bboxes_ignore (list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
                Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # add batch_input_shape in img_metas


        x = self.extract_feat(img)
        losses = self.panoptic_head.forward_train(x, img_metas, gt_bboxes,
                                                  gt_labels, gt_masks,
                                                  gt_semantic_seg,
                                                  gt_bboxes_ignore)

        return losses




