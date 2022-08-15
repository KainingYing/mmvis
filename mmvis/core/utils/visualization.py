# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import random
import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import seaborn as sns
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from mmcv.utils import mkdir_or_exist
from mmdet.core.mask.structures import bitmap_to_polygon


def _get_adaptive_scales(areas, min_area=800, max_area=30000):
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``'min_area'``, the scale is 0.5 while the area is larger than
    ``'max_area'``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Default: 800.
        max_area (int): Upper bound areas for adaptive scales.
            Default: 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


def random_color(seed):
    """Random a color according to the input seed."""
    random.seed(seed)
    colors = sns.color_palette()
    color = random.choice(colors)
    return color


def imshow_tracks(*args, backend='cv2', **kwargs):
    """Show the tracks on the input image."""
    if backend == 'cv2':
        return _cv2_show_tracks(*args, **kwargs)
    elif backend == 'plt':
        return _plt_show_tracks(*args, **kwargs)
    else:
        raise NotImplementedError()


def _gt_show_tracks(img,
                    bboxes,
                    labels,
                    ids,
                    masks=None,
                    classes=None,
                    score_thr=0.0,
                    thickness=2,
                    font_scale=0.4,
                    show=False,
                    wait_time=0,
                    out_file=None):
    """Show the tracks with opencv."""
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert ids.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    # assert bboxes.shape[1] == 5
    if isinstance(img, str):
        img = mmcv.imread(img)

    img_shape = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    # # inds = np.where(bboxes[:, -1] > score_thr)[0]
    # bboxes = bboxes[inds]
    # labels = labels[inds]
    # ids = ids[inds]
    if masks is not None:
        assert masks.ndim == 3
        # masks = masks[inds]
        assert masks.shape[0] == bboxes.shape[0]

    text_width, text_height = 9, 13
    for i, (bbox, label, id) in enumerate(zip(bboxes, labels, ids)):
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        score = float(bbox[-1])

        # bbox
        bbox_color = random_color(id)
        bbox_color = [int(255 * _c) for _c in bbox_color][::-1]
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

        # score
        text = '{:.02f}'.format(score)
        if classes is not None:
            text += f'|{classes[label]}'
        width = len(text) * text_width
        img[y1:y1 + text_height, x1:x1 + width, :] = bbox_color
        cv2.putText(img,
                    text, (x1, y1 + text_height - 2),
                    cv2.FONT_HERSHEY_COMPLEX,
                    font_scale,
                    color=(0, 0, 0))

        # id
        text = str(id)
        width = len(text) * text_width
        img[y1 + text_height:y1 + 2 * text_height,
            x1:x1 + width, :] = bbox_color
        cv2.putText(img,
                    str(id), (x1, y1 + 2 * text_height - 2),
                    cv2.FONT_HERSHEY_COMPLEX,
                    font_scale,
                    color=(0, 0, 0))

        # mask
        if masks is not None:
            mask = masks[i].astype(bool)
            mask_color = np.array(bbox_color, dtype=np.uint8).reshape(1, -1)
            img[mask] = img[mask] * 0.5 + mask_color * 0.5

    if show:
        mmcv.imshow(img, wait_time=wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img


def _cv2_show_tracks(img,
                     bboxes,
                     labels,
                     ids,
                     masks=None,
                     classes=None,
                     score_thr=0.0,
                     thickness=2,
                     font_scale=0.4,
                     show=False,
                     wait_time=0,
                     out_file=None):
    """Show the tracks with opencv."""
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert ids.ndim == 1
    assert bboxes.shape[0] == labels.shape[0] == masks.shape[0] == ids.shape[0]
    assert bboxes.shape[1] == 5 or bboxes.shape[1] == 4
    if isinstance(img, str):
        img = mmcv.imread(img)

    img_shape = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    if bboxes.shape[1] == 5:
        inds = np.where(bboxes[:, -1] > score_thr)[0]
        bboxes = bboxes[inds]
        labels = labels[inds]
        ids = ids[inds]

        assert masks.ndim == 3
        masks = masks[inds]
        assert masks.shape[0] == bboxes.shape[0]

    text_width, text_height = 9, 13
    for i, (bbox, label, id) in enumerate(zip(bboxes, labels, ids)):
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        score = float(bbox[-1])

        # bbox
        bbox_color = random_color(id)
        bbox_color = [int(255 * _c) for _c in bbox_color][::-1]
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

        # score
        text = '{:.02f}'.format(score)
        if classes is not None:
            text += f'|{classes[label]}'
        width = len(text) * text_width
        img[y1:y1 + text_height, x1:x1 + width, :] = bbox_color
        cv2.putText(img,
                    text, (x1, y1 + text_height - 2),
                    cv2.FONT_HERSHEY_COMPLEX,
                    font_scale,
                    color=(0, 0, 0))

        # id
        text = str(id)
        width = len(text) * text_width
        img[y1 + text_height:y1 + 2 * text_height,
            x1:x1 + width, :] = bbox_color
        cv2.putText(img,
                    str(id), (x1, y1 + 2 * text_height - 2),
                    cv2.FONT_HERSHEY_COMPLEX,
                    font_scale,
                    color=(0, 0, 0))

        # mask
        if masks is not None:
            mask = masks[i].astype(bool)
            mask_color = np.array(bbox_color, dtype=np.uint8).reshape(1, -1)
            img[mask] = img[mask] * 0.5 + mask_color * 0.5

    if show:
        mmcv.imshow(img, wait_time=wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img


EPS = 1e-2


def _get_bias_color(base, max_dist=30):
    """Get different colors for each masks.

    Get different colors for each masks by adding a bias
    color to the base category color.
    Args:
        base (ndarray): The base category color with the shape
            of (3, ).
        max_dist (int): The max distance of bias. Default: 30.

    Returns:
        ndarray: The new color for a mask with the shape of (3, ).
    """
    new_color = base + np.random.randint(
        low=-max_dist, high=max_dist + 1, size=3)
    return np.clip(new_color, 0, 255, new_color)


def draw_bboxes(ax, bboxes, color='g', alpha=0.8, thickness=2):
    """Draw bounding boxes on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        bboxes (ndarray): The input bounding boxes with the shape
            of (n, 4).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.

    Returns:
        matplotlib.Axes: The result axes.
    """
    polygons = []
    for i, bbox in enumerate(bboxes):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
    p = PatchCollection(polygons,
                        facecolor='none',
                        edgecolors=color,
                        linewidths=thickness,
                        alpha=alpha)
    ax.add_collection(p)

    return ax


def draw_labels(ax,
                labels,
                ids,
                positions,
                scores=None,
                class_names=None,
                color='w',
                font_size=8,
                scales=None,
                horizontal_alignment='left'):
    """Draw labels on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        labels (ndarray): The labels with the shape of (n, ).
        positions (ndarray): The positions to draw each labels.
        scores (ndarray): The scores for each labels.
        class_names (list[str]): The class names.
        color (list[tuple] | matplotlib.color): The colors for labels.
        font_size (int): Font size of texts. Default: 8.
        scales (list[float]): Scales of texts. Default: None.
        horizontal_alignment (str): The horizontal alignment method of
            texts. Default: 'left'.

    Returns:
        matplotlib.Axes: The result axes.
    """
    for i, (pos, label) in enumerate(zip(positions, labels)):
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        if ids is not None:
            label_text = f'ID: {ids[i]} ' + label_text

        if scores is not None:
            label_text += f'|{scores[i]:.02f}'
        text_color = color[i] if isinstance(color, list) else color

        font_size_mask = font_size if scales is None else font_size * scales[i]
        ax.text(pos[0],
                pos[1],
                f'{label_text}',
                bbox={
                    'facecolor': 'black',
                    'alpha': 0.8,
                    'pad': 0.7,
                    'edgecolor': 'none'
                },
                color=text_color,
                fontsize=font_size_mask,
                verticalalignment='top',
                horizontalalignment=horizontal_alignment)

    return ax


def draw_masks(ax, img, masks, color=None, with_edge=True, alpha=0.8):
    """Draw masks on the image and their edges on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        img (ndarray): The image with the shape of (3, h, w).
        masks (ndarray): The masks with the shape of (n, h, w).
        color (ndarray): The colors for each masks with the shape
            of (n, 3).
        with_edge (bool): Whether to draw edges. Default: True.
        alpha (float): Transparency of bounding boxes. Default: 0.8.

    Returns:
        matplotlib.Axes: The result axes.
        ndarray: The result image.
    """
    taken_colors = set([0, 0, 0])
    if color is None:
        random_colors = np.random.randint(0, 255, (masks.size(0), 3))
        color = [tuple(c) for c in random_colors]
        color = np.array(color, dtype=np.uint8)
    polygons = []
    for i, mask in enumerate(masks):
        if with_edge:
            contours, _ = bitmap_to_polygon(mask)
            polygons += [Polygon(c) for c in contours]

        color_mask = color[i]
        # while tuple(color_mask) in taken_colors:
        #     color_mask = _get_bias_color(color_mask)
        taken_colors.add(tuple(color_mask))

        mask = mask.astype(bool)
        img[mask] = img[mask] * (1 -
                                 alpha) + np.array(color_mask) * 256 * alpha

    p = PatchCollection(polygons,
                        facecolor='none',
                        edgecolors='w',
                        linewidths=1,
                        alpha=0.8)
    ax.add_collection(p)

    return ax, img


def _plt_show_tracks(img,
                     bboxes,
                     labels,
                     ids,
                     masks=None,
                     classes=None,
                     score_thr=0.0,
                     thickness=2,
                     font_scale=13,
                     show=False,
                     wait_time=0,
                     out_file=None,
                     win_name=''):
    """Show the tracks with matplotlib."""
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert ids.ndim == 1
    try:
        assert bboxes.shape[0] == labels.shape[0] == masks.shape[
            0] == ids.shape[0]
    except:
        print(123)
    assert bboxes.shape[1] == 5 or bboxes.shape[1] == 4

    img = mmcv.imread(img).astype(np.uint8)
    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    colors = [random_color(id_) for id_ in ids]

    if bboxes is not None:
        num_bboxes = bboxes.shape[0]

        # bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
        # colors = [bbox_palette[label] for label in labels[:num_bboxes]]
        draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)

        horizontal_alignment = 'left'
        positions = bboxes[:, :2].astype(np.int32) + thickness
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas)
        scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
        draw_labels(ax,
                    labels[:num_bboxes],
                    ids,
                    positions,
                    scores=scores,
                    class_names=classes,
                    color=colors,
                    font_size=font_scale,
                    scales=scales,
                    horizontal_alignment=horizontal_alignment)

    if masks is not None:
        draw_masks(ax, img, masks, colors, with_edge=True)

    plt.imshow(img)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    if sys.platform == 'darwin':
        width, height = canvas.get_width_height(physical=True)
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img


def imshow_mot_errors(*args, backend='cv2', **kwargs):
    """Show the wrong tracks on the input image.

    Args:
        backend (str, optional): Backend of visualization.
            Defaults to 'cv2'.
    """
    if backend == 'cv2':
        return _cv2_show_wrong_tracks(*args, **kwargs)
    elif backend == 'plt':
        return _plt_show_wrong_tracks(*args, **kwargs)
    else:
        raise NotImplementedError()


def _cv2_show_wrong_tracks(img,
                           bboxes,
                           ids,
                           error_types,
                           thickness=2,
                           font_scale=0.4,
                           text_width=10,
                           text_height=15,
                           show=False,
                           wait_time=100,
                           out_file=None):
    """Show the wrong tracks with opencv.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): A ndarray of shape (k, 5).
        ids (ndarray): A ndarray of shape (k, ).
        error_types (ndarray): A ndarray of shape (k, ), where 0 denotes
            false positives, 1 denotes false negative and 2 denotes ID switch.
        thickness (int, optional): Thickness of lines.
            Defaults to 2.
        font_scale (float, optional): Font scale to draw id and score.
            Defaults to 0.4.
        text_width (int, optional): Width to draw id and score.
            Defaults to 10.
        text_height (int, optional): Height to draw id and score.
            Defaults to 15.
        show (bool, optional): Whether to show the image on the fly.
            Defaults to False.
        wait_time (int, optional): Value of waitKey param.
            Defaults to 100.
        out_file (str, optional): The filename to write the image.
            Defaults to None.

    Returns:
        ndarray: Visualized image.
    """
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert ids.ndim == 1, \
        f' ids ndim should be 1, but its ndim is {ids.ndim}.'
    assert error_types.ndim == 1, \
        f' error_types ndim should be 1, but its ndim is {error_types.ndim}.'
    assert bboxes.shape[0] == ids.shape[0], \
        'bboxes.shape[0] and ids.shape[0] should have the same length.'
    assert bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 5, but its {bboxes.shape[1]}.'

    bbox_colors = sns.color_palette()
    # red, yellow, blue
    bbox_colors = [bbox_colors[3], bbox_colors[1], bbox_colors[0]]
    bbox_colors = [[int(255 * _c) for _c in bbox_color][::-1]
                   for bbox_color in bbox_colors]

    if isinstance(img, str):
        img = mmcv.imread(img)
    else:
        assert img.ndim == 3

    img_shape = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    for bbox, error_type, id in zip(bboxes, error_types, ids):
        x1, y1, x2, y2 = bbox[:4].astype(np.int32)
        score = float(bbox[-1])

        # bbox
        bbox_color = bbox_colors[error_type]
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, thickness=thickness)

        # FN does not have id and score
        if error_type == 1:
            continue

        # score
        text = '{:.02f}'.format(score)
        width = (len(text) - 1) * text_width
        img[y1:y1 + text_height, x1:x1 + width, :] = bbox_color
        cv2.putText(img,
                    text, (x1, y1 + text_height - 2),
                    cv2.FONT_HERSHEY_COMPLEX,
                    font_scale,
                    color=(0, 0, 0))

        # id
        text = str(id)
        width = len(text) * text_width
        img[y1 + text_height:y1 + text_height * 2,
            x1:x1 + width, :] = bbox_color
        cv2.putText(img,
                    str(id), (x1, y1 + text_height * 2 - 2),
                    cv2.FONT_HERSHEY_COMPLEX,
                    font_scale,
                    color=(0, 0, 0))

    if show:
        mmcv.imshow(img, wait_time=wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    return img


def _plt_show_wrong_tracks(img,
                           bboxes,
                           ids,
                           error_types,
                           thickness=0.1,
                           font_scale=3,
                           text_width=8,
                           text_height=13,
                           show=False,
                           wait_time=100,
                           out_file=None):
    """Show the wrong tracks with matplotlib.

    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): A ndarray of shape (k, 5).
        ids (ndarray): A ndarray of shape (k, ).
        error_types (ndarray): A ndarray of shape (k, ), where 0 denotes
            false positives, 1 denotes false negative and 2 denotes ID switch.
        thickness (float, optional): Thickness of lines.
            Defaults to 0.1.
        font_scale (float, optional): Font scale to draw id and score.
            Defaults to 3.
        text_width (int, optional): Width to draw id and score.
            Defaults to 8.
        text_height (int, optional): Height to draw id and score.
            Defaults to 13.
        show (bool, optional): Whether to show the image on the fly.
            Defaults to False.
        wait_time (int, optional): Value of waitKey param.
            Defaults to 100.
        out_file (str, optional): The filename to write the image.
            Defaults to None.

    Returns:
        ndarray: Original image.
    """
    assert bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert ids.ndim == 1, \
        f' ids ndim should be 1, but its ndim is {ids.ndim}.'
    assert error_types.ndim == 1, \
        f' error_types ndim should be 1, but its ndim is {error_types.ndim}.'
    assert bboxes.shape[0] == ids.shape[0], \
        'bboxes.shape[0] and ids.shape[0] should have the same length.'
    assert bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 5, but its {bboxes.shape[1]}.'

    bbox_colors = sns.color_palette()
    # red, yellow, blue
    bbox_colors = [bbox_colors[3], bbox_colors[1], bbox_colors[0]]

    if isinstance(img, str):
        img = plt.imread(img)
    else:
        assert img.ndim == 3
        img = mmcv.bgr2rgb(img)

    img_shape = img.shape
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])

    plt.imshow(img)
    plt.gca().set_axis_off()
    plt.autoscale(False)
    plt.subplots_adjust(top=1,
                        bottom=0,
                        right=1,
                        left=0,
                        hspace=None,
                        wspace=None)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.rcParams['figure.figsize'] = img_shape[1], img_shape[0]

    for bbox, error_type, id in zip(bboxes, error_types, ids):
        x1, y1, x2, y2, score = bbox
        w, h = int(x2 - x1), int(y2 - y1)
        left_top = (int(x1), int(y1))

        # bbox
        plt.gca().add_patch(
            Rectangle(left_top,
                      w,
                      h,
                      thickness,
                      edgecolor=bbox_colors[error_type],
                      facecolor='none'))

        # FN does not have id and score
        if error_type == 1:
            continue

        # score
        text = '{:.02f}'.format(score)
        width = len(text) * text_width
        plt.gca().add_patch(
            Rectangle((left_top[0], left_top[1]),
                      width,
                      text_height,
                      thickness,
                      edgecolor=bbox_colors[error_type],
                      facecolor=bbox_colors[error_type]))

        plt.text(left_top[0],
                 left_top[1] + text_height + 2,
                 text,
                 fontsize=font_scale)

        # id
        text = str(id)
        width = len(text) * text_width
        plt.gca().add_patch(
            Rectangle((left_top[0], left_top[1] + text_height + 1),
                      width,
                      text_height,
                      thickness,
                      edgecolor=bbox_colors[error_type],
                      facecolor=bbox_colors[error_type]))
        plt.text(left_top[0],
                 left_top[1] + 2 * (text_height + 1),
                 text,
                 fontsize=font_scale)

    if out_file is not None:
        mkdir_or_exist(osp.abspath(osp.dirname(out_file)))
        plt.savefig(out_file, dpi=300, bbox_inches='tight', pad_inches=0.0)

    if show:
        plt.draw()
        plt.pause(wait_time / 1000.)

    plt.clf()
    return img


# def frames2video(frame_dir: str,
#                  video_file: str,
#                  fps: float = 30,
#                  fourcc: str = 'XVID',
#                  filename_tmpl: str = '{:06d}.jpg',
#                  start: int = 0,
#                  end: int = 0,
#                  show_progress: bool = True) -> None:
#     """Read the frame images from a directory or numpy and join them as a video.
#
#     Args:
#         frame_dir (str): The directory containing video frames.
#         video_file (str): Output filename.
#         fps (float): FPS of the output video.
#         fourcc (str): Fourcc of the output video, this should be compatible
#             with the output file type.
#         filename_tmpl (str): Filename template with the index as the variable.
#         start (int): Starting frame index.
#         end (int): Ending frame index.
#         show_progress (bool): Whether to show a progress bar.
#     """
#     if end == 0:
#         ext = filename_tmpl.split('.')[-1]
#         end = len([name for name in scandir(frame_dir, ext)])
#     first_file = osp.join(frame_dir, filename_tmpl.format(start))
#     check_file_exist(first_file, 'The start frame not found: ' + first_file)
#     img = cv2.imread(first_file)
#     height, width = img.shape[:2]
#     resolution = (width, height)
#     vwriter = cv2.VideoWriter(video_file, VideoWriter_fourcc(*fourcc), fps,
#                               resolution)
#
#     def write_frame(file_idx):
#         filename = osp.join(frame_dir, filename_tmpl.format(file_idx))
#         img = cv2.imread(filename)
#         vwriter.write(img)
#
#     if show_progress:
#         track_progress(write_frame, range(start, end))
#     else:
#         for i in range(start, end):
#             write_frame(i)
#     vwriter.release()
