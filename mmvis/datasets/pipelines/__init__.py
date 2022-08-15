# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.builder import PIPELINES

from .auto_augment import (AutoAugment, BrightnessTransform, COCO2Seq,
                           ColorTransform, ContrastTransform,
                           EqualizeTransform, Rotate, Shear, Translate)
from .compose import Compose
from .formatting import (CheckPadMaskValidity, ConcatSameTypeFrames,
                         ConcatVideoReferences, ReIDFormatBundle,
                         SeqDefaultFormatBundle, SeqRename, ToList,
                         VideoCollect)
from .loading import LoadMultiImagesFromFile, SeqLoadAnnotations
from .processing import MatchInstances, PairSampling, TridentSampling
from .transforms import (SeqBboxJitter, SeqBlurAug, SeqBrightnessAug,
                         SeqColorAug, SeqCropLikeSiamFC, SeqCropLikeStark,
                         SeqGrayAug, SeqNormalize, SeqPad,
                         SeqPhotoMetricDistortion, SeqRandomCrop,
                         SeqRandomFlip, SeqResize, SeqShiftScaleAug)

__all__ = [
    'PIPELINES', 'LoadMultiImagesFromFile', 'SeqLoadAnnotations', 'SeqResize',
    'SeqNormalize', 'SeqRandomFlip', 'SeqPad', 'SeqDefaultFormatBundle',
    'VideoCollect', 'CheckPadMaskValidity', 'ConcatVideoReferences',
    'MatchInstances', 'SeqRandomCrop', 'SeqPhotoMetricDistortion',
    'SeqCropLikeSiamFC', 'SeqShiftScaleAug', 'SeqBlurAug', 'SeqColorAug',
    'ToList', 'ReIDFormatBundle', 'SeqGrayAug', 'SeqBrightnessAug',
    'SeqBboxJitter', 'SeqCropLikeStark', 'TridentSampling',
    'ConcatSameTypeFrames', 'PairSampling', 'SeqRename', 'AutoAugment',
    'Compose', 'COCO2Seq'
]
