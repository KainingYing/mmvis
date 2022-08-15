import argparse

import mmcv
from mmcv.runner.checkpoint import load_from_http

CHECKPOINT2URL = {
    'mask2former': {
        'resnet50':
        'https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_r50_lsj_'
        '8x2_50e_coco/mask2former_r50_lsj_8x2_50e_coco_20220506_191028-8e96e88b.pth',
        'resnet101':
        'https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_r101_lsj_'
        '8x2_50e_coco/mask2former_r101_lsj_8x2_50e_coco_20220426_100250-c50b6fa6.pth',
        'swim-t':
        'https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-t-p4-w7-224_lsj_'
        '8x2_50e_coco/mask2former_swin-t-p4-w7-224_lsj_8x2_50e_coco_20220508_091649-4a943037.pth',
        'swim-s':
        'https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_'
        '8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth'
    }
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert checkpoint key-value pairs')
    parser.add_argument('--model',
                        type=str,
                        help='the model name',
                        default='mask2former')
    parser.add_argument('--backbone',
                        type=str,
                        help='the backbone of model',
                        default='resnet50')
    parser.add_argument('--save-dir',
                        type=str,
                        help='the dir to save converted checkpoint',
                        default='checkpoint/')
    args = parser.parse_args()
    return args


def convert_mask2former(checkpoint):
    state_dict = checkpoint['state_dict']

    convert_dict = {'panoptic_head': 'head'}
    state_dict_keys = list(state_dict.keys())

    for k in state_dict_keys:
        for ori_key, convert_key in convert_dict.items():
            if ori_key in k:
                convert_key = k.replace(ori_key, convert_key)
                state_dict[convert_key] = state_dict[k]
                del state_dict[k]
    return checkpoint


def main(args):
    checkpoint = load_from_http(CHECKPOINT2URL[args.model][args.backbone])

    checkpoint = convert_mask2former(checkpoint)

    mmcv.mkdir_or_exist(args.save_dir)

    pass


if __name__ == '__main__':
    args = parse_args()
    main(args)
