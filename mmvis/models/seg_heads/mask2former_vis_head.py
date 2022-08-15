# Copyright (c) ZJUTCV. All rights reserved.
import copy
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.ops import point_sample
from mmcv.runner import BaseModule, ModuleList, force_fp32
from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.core.mask import mask2bbox
from mmdet.models.utils import get_uncertain_point_coords_with_randomness

from mmvis.utils import AvoidCUDAOOM, AvoidOOM, profile_time

from ..builder import HEADS, build_loss


@HEADS.register_module()
class Mask2FormerVISHead(BaseModule):
    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_classes,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers

        assert pixel_decoder.encoder.transformerlayers.attn_cfgs.num_levels == num_transformer_feat_level

        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(in_channels=in_channels,
                              feat_channels=feat_channels,
                              out_channels=out_channels)

        self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(feat_channels,
                           self.decoder_embed_dims,
                           kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = build_positional_encoding(
            positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        # version = local_metadata.get('version', None)
        # if (version is None or version < 2) and self.__class__ is DETRHead:
        convert_dict = {'panoptic_head': 'head'}
        state_dict_keys = list(state_dict.keys())
        for k in state_dict_keys:
            for ori_key, convert_key in convert_dict.items():
                if ori_key in k:
                    convert_key = k.replace(ori_key, convert_key)
                    state_dict[convert_key] = state_dict[k]
                    del state_dict[k]

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def forward_train(self, feats, video_metas, gt_bboxes, gt_labels, gt_masks,
                      gt_instance_ids):

        # forward
        # with profile_time('head_forward', 'head_forward'):
        all_cls_scores, all_mask_preds = self(feats, video_metas)

        gt_labels, gt_masks = self.preprocess_gt(gt_labels, gt_masks,
                                                 gt_instance_ids, video_metas)

        # loss
        # with profile_time('loss', 'loss'):
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks,
                           video_metas)

        return losses

    def forward(self, feats, img_metas):

        batch_size = len(img_metas)
        num_frames = len(img_metas[0])

        # todo: we add mask here padding 第一处

        flatten_img_metas = list(itertools.chain.from_iterable(img_metas))
        num_imgs = len(flatten_img_metas)
        input_img_h, input_img_w = flatten_img_metas[0]['batch_input_shape']
        img_masks = feats[0].new_ones((num_imgs, input_img_h, input_img_w))

        for img_id in range(num_imgs):
            img_h, img_w, _ = flatten_img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        for feat in feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))

        mask_features, multi_scale_memorys = self.pixel_decoder(
            feats, mlvl_masks)
        # multi_scale_memorys (from low resolution to high resolution)

        # todo: video instance segmentation
        mask_features = mask_features.view(batch_size, num_frames,
                                           mask_features.shape[-3],
                                           mask_features.shape[-2],
                                           mask_features.shape[-1])

        # bs = bst // self.num_frames if self.training else 1
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            h, w = decoder_input.shape[-2:]
            # todo: 修改了两处, padding第二处
            mask = mlvl_masks[len(mlvl_masks) - i - 1]
            decoder_positional_encoding = self.decoder_positional_encoding(
                decoder_input.view(batch_size, num_frames, -1, h, w),
                mask.view(batch_size, num_frames, h, w)).flatten(3)
            # decoder_positional_encoding = self.decoder_positional_encoding(
            #     decoder_input.view(batch_size, num_frames, -1, h, w), None).flatten(3)
            decoder_positional_encoding = decoder_positional_encoding. \
                view(batch_size, num_frames, -1, h * w).permute(1, 3, 0, 2).flatten(0, 1)

            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_input = decoder_input.flatten(
                2) + self.level_embed.weight[i][None, ..., None]
            decoder_input = decoder_input.view(batch_size, num_frames, -1,
                                               h * w).permute(1, 3, 0,
                                                              2).flatten(0, 1)

            # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            # todo: mask这个部分需要注意  mask这个部分是全0，未来可能需要考虑padding部分
            # mask = decoder_input.new_zeros(
            #     (batch_size, ) + multi_scale_memorys[i].shape[-2:],
            #     dtype=torch.bool)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(
            (1, batch_size, 1))

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self.forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(
                attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)  # todo 这个地方需要加入padding，这是一个bug
            cls_pred, mask_pred, attn_mask = self.forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries,h, w).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (num_queries, batch_size, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,btchw->bqthw', mask_embed, mask_feature)
        b, q, t, _, _ = mask_pred.shape

        # NOTE: prediction is of higher-resolution
        # [B, Q, T, H, W] -> [B, Q, T*H*W] -> [B, h, Q, T*H*W] -> [B*h, Q, T*HW]
        attn_mask = F.interpolate(mask_pred.flatten(0, 1),
                                  size=attn_mask_target_size,
                                  mode='bilinear',
                                  align_corners=False).view(
                                      b, q, t, attn_mask_target_size[0],
                                      attn_mask_target_size[1])

        # attn_mask = F.interpolate(
        #     mask_pred,
        #     attn_mask_target_size,
        #     mode='bilinear',
        #     align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(
            1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def preprocess_gt(self, gt_labels_list, gt_masks_list,
                      gt_instance_ids_list, video_metas):

        num_classes_list = [self.num_classes] * len(gt_labels_list)

        targets = multi_apply(self.preprocess_vis_gt, gt_labels_list,
                              gt_masks_list, gt_instance_ids_list,
                              num_classes_list, video_metas)
        labels, masks = targets
        return labels, masks

    def preprocess_vis_gt(self, gt_labels, gt_masks, gt_instance_ids,
                          num_classes, video_metas):

        # instance_ids_set = set(torch.cat(gt_instance_ids).detach().cpu().numpy().tolist())
        # gt_label_seq = torch.cat(gt_labels, axis=0)
        # 需要看看是不是空的
        gt_masks_tensor = [
            gt_mask.pad(video_metas[0]['batch_input_shape'][:2],
                        pad_val=0).to_tensor(dtype=torch.bool,
                                             device=gt_labels[0].device)
            for gt_mask in gt_masks
        ]
        num_frames = len(gt_labels)
        gt_instance_ids_seq = torch.cat(gt_instance_ids, axis=0)

        gt_instance_ids_unique = gt_instance_ids_seq.unique()

        num_instances = gt_instance_ids_unique.numel()

        gt_frame_id = torch.cat([
            gt_instance_ids_unique.new_full((instance_id.shape[0], ), frame_id)
            for frame_id, instance_id in enumerate(gt_instance_ids)
        ])
        gt_labels_seq = torch.cat(gt_labels)
        gt_masks_seq = torch.cat(gt_masks_tensor, dim=0)

        labels = []
        masks = []
        for i in range(num_instances):
            index = gt_instance_ids_seq == gt_instance_ids_unique[i]

            label = gt_labels_seq[index].unique()
            mask = gt_masks_seq.new_zeros(
                (num_frames, gt_masks_seq.shape[-2], gt_masks_seq.shape[-1]),
                dtype=torch.bool)

            mask[gt_frame_id[index]] = gt_masks_seq[index]
            labels.append(label)
            masks.append(mask)

        labels = torch.cat(labels, 0)
        masks = torch.stack(masks, 0)
        # if len(labels) != 0:
        #     labels = torch.cat(labels, 0)
        #     masks = torch.stack(masks, 0)
        # else:
        #     labels = m

        return labels, masks

    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list,
             gt_masks_list, img_metas):
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]

        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_mask, losses_dice = multi_apply(
            self.loss_single, all_cls_scores, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, img_metas_list)

        # # 不进行拷贝
        # losses_cls, losses_mask, losses_dice = multi_apply(
        #     self.loss_single,
        #     all_cls_scores,
        #     all_mask_preds,
        #     gt_labels_list=gt_labels_list,
        #     gt_masks_list=gt_masks_list,
        #     img_metas=img_metas)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, img_metas):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
                                           gt_labels_list, gt_masks_list,
                                           img_metas)

        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 label_weights,
                                 avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        mask_preds = mask_preds.flatten(0, 1)[:, None]
        mask_targets = mask_targets.flatten(0, 1)[:, None]

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_preds, None, self.num_points, self.oversample_ratio,
                self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(mask_targets.float(),
                                              points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(mask_preds, points_coords).squeeze(1)

        # mask_point_preds = mask_point_preds[:, ]

        # dice loss
        loss_dice = self.loss_dice(mask_point_preds,
                                   mask_point_targets,
                                   avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        loss_mask = self.loss_mask(mask_point_preds,
                                   mask_point_targets,
                                   avg_factor=num_total_masks *
                                   self.num_points)

        return loss_cls, loss_mask, loss_dice

    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, img_metas):
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      mask_preds_list, gt_labels_list,
                                      gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks,
                           img_metas):
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2),
                                  device=cls_score.device)
        # shape (num_queries, num_points)
        mask_points_pred = point_sample(mask_pred,
                                        point_coords.repeat(num_queries, 1,
                                                            1)).flatten(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(gt_masks.float(),
                                       point_coords.repeat(num_gts, 1,
                                                           1)).flatten(1)

        # assign and sample
        assign_result = self.assigner.assign(cls_score, mask_points_pred,
                                             gt_labels, gt_points_masks,
                                             img_metas)
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((self.num_queries, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((self.num_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds,
                neg_inds)

    # def simple_test(self,
    #                 feats,
    #                 video_metas,
    #                 rescale,
    #                 **kwargs):
    #     # todo: in this test stage, we need to save the consumed memory and speed up.
    #
    #     all_cls_scores, all_mask_preds = self(feats, video_metas)
    #
    #     mask_cls_results = all_cls_scores[-1]
    #     mask_pred_results = all_mask_preds[-1]
    #
    #     # mask_cls_results = mask_cls_results.cpu()
    #     # mask_pred_results = mask_pred_results.cpu()
    #
    #     # 在这里的bs只是1，所以没必要多余的循环了
    #
    #
    #     img_shape = video_metas[0][0]['batch_input_shape']
    #
    #     batch_size, num_queries, num_frames = mask_pred_results.shape[:3]
    #
    #     mask_pred_results = mask_pred_results.flatten(0, 1)
    #
    #     mask_pred_results = F.interpolate(
    #         mask_pred_results,
    #         size=(img_shape[0], img_shape[1]),
    #         mode='bilinear',
    #         align_corners=False)
    #     mask_pred_results = mask_pred_results.reshape(batch_size, num_queries, num_frames, *img_shape)
    #     mask_pred_result = mask_pred_results[0]
    #     mask_cls_result = mask_cls_results[0]
    #     meta = video_metas[0]
    #
    #     results = []
    #     # for mask_cls_result, mask_pred_result, meta in zip(
    #     #         mask_cls_results, mask_pred_results, video_metas):
    #     # remove padding
    #     img_height, img_width = meta[0]['img_shape'][:2]
    #     mask_pred_result = mask_pred_result[:, :, :img_height, :img_width]
    #
    #     if rescale:
    #         pass
    #         # return result in original resolution
    #         # ori_height, ori_width = meta[0]['ori_shape'][:2]
    #         # mask_pred_result = F.interpolate(
    #         #     mask_pred_result,
    #         #     size=(ori_height, ori_width),
    #         #     mode='bilinear',
    #         #     align_corners=False)
    #
    #     result = dict()
    #     try:
    #         ins_results = self.postprocess(mask_cls_result, mask_pred_result, meta)
    #
    #     except RuntimeError as e:
    #         # NOTE: the string may change?
    #         # if 'CUDA out of memory. ' in str(e):
    #         #     mask_pred_result = mask_pred_results.cpu()[0]
    #         #     mask_cls_result = mask_cls_results.cpu()[0]
    #         #     ins_results = self.postprocess(mask_cls_result, mask_pred_result, meta)
    #         #
    #         # else:
    #         raise
    #     result['ins_results'] = ins_results
    #     results.append(result)
    #
    #     return results
    #
    # # @AvoidCUDAOOM.retry_if_cuda_oom
    # def postprocess(self,
    #                 mask_cls,
    #                 mask_pred,
    #                 meta):
    #     # torch.cuda.empty_cache()
    #     # mask_cls.cpu()
    #     # mask_pred.cpu()
    #     max_per_image = self.test_cfg.get('max_per_image', 10)
    #     num_queries = mask_cls.shape[0]
    #     # shape (num_queries, num_class)
    #     scores = F.softmax(mask_cls, dim=-1)[:, :-1]
    #     # shape (num_queries * num_class, )
    #     labels = torch.arange(self.num_classes, device=mask_cls.device). \
    #         unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
    #     scores_per_image, top_indices = scores.flatten(0, 1).topk(
    #         max_per_image, sorted=False)
    #     labels_per_image = labels[top_indices]
    #
    #     query_indices = top_indices // self.num_classes
    #     mask_pred = mask_pred[query_indices]
    #
    #     ori_height, ori_width = meta[0]['ori_shape'][:2]
    #     mask_pred = F.interpolate(
    #         mask_pred,
    #         size=(ori_height, ori_width),
    #         mode='bilinear',
    #         align_corners=False)
    #     mask_pred_binary = (mask_pred > 0).float()
    #
    #     # mask_scores_per_image = scores_per_image.new_tensor([1.])
    #     # with _ignore_torch_cuda_oom():
    #     mask_scores_per_image = (mask_pred.sigmoid() *
    #                              mask_pred_binary).flatten(1).sum(1) / (
    #                                     mask_pred_binary.flatten(1).sum(1) + 1e-6)  # this operation will bring gains
    #     det_scores = scores_per_image * mask_scores_per_image
    #     mask_pred_binary = mask_pred_binary.bool()
    #     bboxes = mask2bbox(mask_pred_binary)
    #     bboxes = torch.cat([bboxes, det_scores[:, None]], dim=-1)
    #
    #     return labels_per_image, bboxes, mask_pred_binary

    def simple_test(self, feats, video_metas, rescale, **kwargs):
        # todo: in this test stage, we need to save the consumed memory and speed up.

        all_cls_scores, all_mask_preds = self(feats, video_metas)

        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        # img_shape = video_metas[0][0]['batch_input_shape']
        #
        # # todo: 这种缩放还是放在最后比较好
        # mask_pred_results = F.interpolate(
        #     mask_pred_results,
        #     size=(img_shape[0], img_shape[1]),
        #     mode='bilinear',
        #     align_corners=False)

        results = []

        for mask_cls_result, mask_pred_result, meta in zip(
                mask_cls_results, mask_pred_results, video_metas):
            # remove padding
            # img_height, img_width = meta['img_shape'][:2]
            # mask_pred_result = mask_pred_result[:, :img_height, :img_width]
            #
            # if rescale:
            #     # return result in original resolution
            #     ori_height, ori_width = meta['ori_shape'][:2]
            #     mask_pred_result = F.interpolate(
            #         mask_pred_result[:, None],
            #         size=(ori_height, ori_width),
            #         mode='bilinear',
            #         align_corners=False)[:, 0]

            result = dict()

            ins_results = self.postprocess(mask_cls_result, mask_pred_result,
                                           meta)
            result['ins_results'] = ins_results

            results.append(result)

        return results

    @AvoidCUDAOOM.retry_if_cuda_oom
    def postprocess(self, mask_cls, mask_pred, meta):

        max_per_image = self.test_cfg.get('max_per_image', 10)
        num_queries = mask_cls.shape[0]
        # shape (num_queries, num_class)
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # shape (num_queries * num_class, )
        labels = torch.arange(self.num_classes, device=mask_cls.device). \
            unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
        scores_per_image, top_indices = scores.flatten(0,
                                                       1).topk(max_per_image,
                                                               sorted=False)
        labels_per_image = labels[top_indices]

        query_indices = top_indices // self.num_classes
        mask_pred = mask_pred[query_indices]

        # resize to batch input image
        batch_input_shape = meta[0]['batch_input_shape']
        # num_queries, num_frames = mask_pred.shape[:2]
        # mask_pred = mask_pred.flatten(0, 1)
        mask_pred = F.interpolate(mask_pred,
                                  size=(batch_input_shape[0],
                                        batch_input_shape[1]),
                                  mode='bilinear',
                                  align_corners=False)

        # remove padding
        img_height, img_width = meta[0]['img_shape'][:2]
        mask_pred = mask_pred[..., :img_height, :img_width]

        # resize to origin image
        ori_height, ori_width = meta[0]['ori_shape'][:2]
        mask_pred = F.interpolate(mask_pred,
                                  size=(ori_height, ori_width),
                                  mode='bilinear',
                                  align_corners=False)

        mask_pred_binary = (mask_pred > 0).float()
        # mask_scores_per_image = scores_per_image.new_tensor([1.])
        # with _ignore_torch_cuda_oom():
        mask_scores_per_image = (mask_pred.sigmoid() *
                                 mask_pred_binary).flatten(1).sum(1) / (
                                     mask_pred_binary.flatten(1).sum(1) + 1e-6
                                 )  # this operation will bring gains
        det_scores = scores_per_image * mask_scores_per_image
        mask_pred_binary = mask_pred_binary.bool()
        bboxes = mask2bbox(mask_pred_binary)
        bboxes = torch.cat([bboxes, det_scores[:, None]], dim=-1)

        return labels_per_image, bboxes, mask_pred_binary
