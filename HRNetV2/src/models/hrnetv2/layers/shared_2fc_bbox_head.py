# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# This file or its part has been derived from the following repository
# and modified: https://github.com/open-mmlab/mmdetection/tree/v2.28.2
# ============================================================================
"""Bounding box head for RCNN models."""
from typing import Sequence, Tuple

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import ops

from src.config import Config


class Shared2FCBBoxHead(nn.Cell):
    """Bounding box head block for RCNN models. Consists of 2 fully connected
    shared layers, classification layer and regression layer.

    Args:
        train_batch_size (int): Train batch size (image num per step).
        test_batch_size (int): Test batch size (image num per step).
        num_gts (int): Num Ground Truth.
        bbox_coder (nn.Cell): Bounding box coder.
        loss_cls (nn.Cell): Classification loss block.
        loss_bbox (nn.Cell): Regression loss block.
        targets_generator (nn.Cell): Block that generates targets. Contain
            assigner, sampler and bbox_coder (encode bounding boxes).
        num_classes (int): number of classification categories
            (excluding background).
        in_channels (int): number of input channels.
        fc_out_channels (int): number of output channels of layer before
            classification.
        roi_feat_size (int): h and w size of input feature maps.
        loss_cls_weight (float): Classification loss weight.
        loss_bbox_weight (float): Regression loss weight.
        train_cfg (Config): Model training configurations.
        test_cfg (Config): Model inference configurations.

    Example:
        >>> class DummyCoder(nn.Cell):
        ...     def __init__(self):
        ...         super(DummyCoder, self).__init__()
        ...     def encode(self, proposals, gt):
        ...         return proposals - gt
        ...     def decode(self, rois, delta):
        ...         return ms.ops.abs(rois + delta)
        >>> coder = DummyCoder()
        >>> loss_cls = nn.SoftmaxCrossEntropyWithLogits(reduction='none')
        >>> loss_bbox = nn.L1Loss(reduction='none')
        >>> bbox_head = Shared2FCBBoxHead(
        ...     bbox_coder=DummyCoder(),
        ...     loss_cls=loss_cls,
        ...     loss_bbox=loss_bbox,
        ...     targets_generator=None,
        ...     num_classes=80,
        ...     roi_feat_size=7,
        ...     in_channels=256,
        ...     test_cfg=Config(
        ...         dict(
        ...             rcnn=dict(
        ...                 score_thr=0.05,
        ...                 iou_threshold=0.5,
        ...                 max_per_img=100
        ...             ),
        ...             rpn=dict(
        ...                 max_per_img=1000
        ...             )
        ...         )
        ...     )
        ... )
        >>> roi_feats = ms.ops.Tensor(
        ...     np.random.randn(1000, 256, 7, 7), ms.float32
        ... )
        >>> cls_logits, reg_logits = bbox_head(roi_feats)
        >>> cls_logits.shape # num_classes + 1
        (1000, 81)
        >>> reg_logits.shape # num_classes * 4
        (1000, 320)
    """

    def __init__(
            self,
            bbox_coder: nn.Cell,
            loss_cls: nn.Cell,
            loss_bbox: nn.Cell,
            targets_generator: nn.Cell,
            train_batch_size: int = 1,
            test_batch_size: int = 1,
            num_gts: int = 100,
            num_classes: int = 80,
            in_channels: int = 256,
            fc_out_channels: int = 1024,
            roi_feat_size: int = 7,
            reg_class_agnostic=False,
            loss_cls_weight: float = 1.0,
            loss_bbox_weight: float = 1.0,
            norm_bboxes=False,
            train_cfg: Config = Config(
                dict(
                    rpn_proposal=dict(max_per_img=1000)
                )
            ),
            test_cfg: Config = Config(
                dict(
                    rcnn=dict(
                        score_thr=0.05,
                        iou_threshold=0.5,
                        max_per_img=100
                    ),
                    rpn=dict(
                        max_per_img=1000
                    )
                )
            )
    ):
        """Init Shared2FCBBoxHead."""
        super().__init__()
        self.ms_type = ms.float32
        self.dtype = np.float32
        self.num_gts = num_gts
        self.in_channels = in_channels
        self.fc_out_channels = fc_out_channels
        self.roi_feat_size = roi_feat_size
        self.num_classes = num_classes
        self.num_cls_bbox = num_classes
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.reg_class_agnostic = reg_class_agnostic
        self.norm_bboxes = norm_bboxes

        self.loss_cls_weight = ms.Tensor(
            np.array(loss_cls_weight).astype(self.dtype)
        )
        self.loss_reg_weight = ms.Tensor(
            np.array(loss_bbox_weight).astype(self.dtype)
        )

        # add shared convs and fcs
        self.shared_fcs, last_layer_dim = (
            self._add_fc_branch(self.in_channels)
        )

        # reconstruct fc_cls and fc_reg since input channels are changed
        self.fc_cls = nn.Dense(last_layer_dim, self.num_classes + 1)

        # reconstruct fc_cls and fc_reg since input channels are changed
        self.fc_reg = nn.Dense(
            last_layer_dim,
            4 if self.reg_class_agnostic else self.num_classes * 4
        )

        self.relu = ops.ReLU()
        self.flatten = ops.Flatten()
        self.loss_cls = loss_cls

        self.loss_bbox = loss_bbox
        self.bbox_coder = bbox_coder

        self.reshape = ops.Reshape()
        self.onehot = ops.OneHot()
        self.squeeze = ops.Squeeze()
        self.softmax = ops.Softmax(axis=1)
        self.split = ops.Split(axis=0, output_num=self.test_batch_size)
        self.split_shape = ops.Split(axis=0, output_num=4)

        self.on_value = ms.Tensor(1.0, ms.float32)
        self.off_value = ms.Tensor(0.0, ms.float32)
        self.value = ms.Tensor(1.0, self.ms_type)
        self.eps = ms.Tensor(1e-10, self.ms_type)

        self.bbox_coder = bbox_coder

        self.targets_generator = targets_generator

        self.rpn_max_num = test_cfg.rpn.max_per_img
        self.max_num = test_cfg.rcnn.max_per_img
        self.score_thr = test_cfg.rcnn.score_thr
        self.iou_threshold = test_cfg.rcnn.iou_threshold

        self.test_topk = ops.TopK(sorted=True)
        self.nms_test = ops.NMSWithMask(self.iou_threshold)

        ones_mask = np.ones((self.rpn_max_num, 1)).astype(np.bool)
        zeros_mask = np.zeros((self.rpn_max_num, 1)).astype(np.bool)
        self.bbox_mask = ms.Tensor(
            np.concatenate(
                (ones_mask, zeros_mask, ones_mask, zeros_mask),
                axis=1
            )
        )

        self.test_score_thresh = ms.Tensor(
            np.ones((self.rpn_max_num, 1))
            .astype(self.dtype) * self.score_thr
        )
        self.test_score_zeros = ms.Tensor(
            np.ones((self.rpn_max_num, 1))
            .astype(self.dtype) * 0
        )
        self.test_box_zeros = ms.Tensor(
            np.ones((self.rpn_max_num, 4))
            .astype(self.dtype) * -1
        )

    def _add_fc_branch(self, in_channels: int):
        """Add shared or separable branch that consists of 2 Dense layers."""
        last_layer_dim = in_channels
        # add branch specific fc layers
        branch_fcs = nn.CellList()
        # for shared branch, only consider self.with_avg_pool
        # for separated branches, also consider self.num_shared_fcs
        last_layer_dim *= (self.roi_feat_size * self.roi_feat_size)
        for i in range(2):
            fc_in_channels = (
                last_layer_dim if i == 0 else self.fc_out_channels)
            branch_fcs.append(
                nn.Dense(fc_in_channels, self.fc_out_channels)
            )
        last_layer_dim = self.fc_out_channels
        return branch_fcs, last_layer_dim

    def construct(self, feature_map: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        """Forward BboxHead.

        Args:
            feature_map (ms.Tensor):  Input features. Shape
                (n, self.in_channels, self.roi_feat_size, self.roi_feat_size).

        Returns:
            ms.Tensor:
                Classification results. Shape (n, self.num_classes + 1).
            ms.Tensor:
                Localization results. Shape (n, self.num_classes * 4).
        """
        x = self.flatten(feature_map)

        for layer in self.shared_fcs:
            x = self.relu(layer(x))

        x_cls = self.fc_cls(x)
        x_reg = self.fc_reg(x)

        return x_cls, x_reg

    def loss(
            self,
            cls_score: ms.Tensor,
            bbox_pred: ms.Tensor,
            bbox_targets: ms.Tensor,
            labels: ms.Tensor,
            weights: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        """Loss method.

        Args:
            cls_score (ms.Tensor): Predictions for classification branch.
                Shape (n, self.num_classes + 1).
            bbox_pred (ms.Tensor): Predictions for detection branch.
                Shape (n, self.num_classes * 4).
            bbox_targets (ms.Tensor): Target regressions. Shape (n, 4).
            labels (ms.Tensor): Target classifications. Shape (n,).
            weights (ms.Tensor): Mask that show what object is valid.
                Shape (n,).

        Returns:
            ms.Tensor: Total loss. Scalar.
            ms.Tensor: Classification loss. Scalar.
            ms.Tensor: Regression loss. Scalar.
        """
        bbox_weights = self.cast(
            ops.logical_and(ops.gt(labels, 0), weights),
            ms.int32
        ) * labels
        labels = self.onehot(labels, self.num_classes + 1, self.on_value,
                             self.off_value)

        loss_cls = self.loss_cls(cls_score, labels)
        weights = self.cast(weights, self.ms_type)
        loss_cls = loss_cls * weights
        loss_cls = ops.reduce_sum(loss_cls, (0,)) / ops.reduce_sum(weights, (0,))

        if self.reg_class_agnostic:
            bbox_weights = self.cast(bbox_weights, self.ms_type)
            bbox_weights = ops.tile(
                ops.expand_dims(bbox_weights, -1), (1, 4)
            )
            pos_bbox_pred = bbox_pred
        else:
            bbox_weights = self.onehot(
                bbox_weights, self.num_classes + 1,
                self.on_value, self.off_value
            )
            bbox_weights = bbox_weights[:, 1:]
            bbox_weights = self.cast(bbox_weights, self.ms_type)
            bbox_weights = ops.tile(
                ops.expand_dims(bbox_weights, -1), (1, 1, 4)
            )
            pos_bbox_pred = self.reshape(bbox_pred, (-1, self.num_cls_bbox, 4))
            bbox_targets = ops.expand_dims(bbox_targets, 1)
            bbox_targets = ops.tile(bbox_targets, (1, self.num_classes, 1))

        loss_reg = self.loss_bbox(pos_bbox_pred, bbox_targets) * bbox_weights
        loss_reg = ops.reduce_sum(loss_reg) / (ops.reduce_sum(weights) + self.eps)

        loss_reg = loss_reg * self.loss_reg_weight
        loss_cls = loss_cls * self.loss_cls_weight
        loss = loss_cls + loss_reg

        return loss, loss_cls, loss_reg

    def get_targets(
            self,
            gt_bboxes: ms.Tensor,
            gt_labels: ms.Tensor,
            gt_valids: ms.Tensor,
            proposal: Sequence[ms.Tensor],
            proposal_mask: Sequence[ms.Tensor]
    ) -> Tuple[
        Sequence[ms.Tensor], Sequence[ms.Tensor], Sequence[ms.Tensor],
        Sequence[ms.Tensor]
    ]:
        """Prepare proposed samples for training.

        Args:
            gt_bboxes (ms.Tensor): Ground truth bounding bboxes. Shape
                (n, self.num_gts, 4).
            gt_labels (ms.Tensor): Ground truth labels for training. Shape
                (n, self.num_gts).
            gt_valids (ms.Tensor): Mask that show what GT bboxes are valid.
                Shape (n, self.num_gts).
            proposal (Sequence[ms.Tensor]): Proposed bounding bboxes. Shape
                (n, self.train_cfg.rpn_proposed.max_per_img).
            proposal_mask (Sequence[ms.Tensor]): Mask that shows what proposed
                bounding boxes are valid. Shape
                (n, self.train_cfg.rpn_proposed.max_per_img).

        Returns:
            Sequence[ms.Tensor]: Sampled bounding boxes.
            Sequence[ms.Tensor]: Localization targets for each bbox.
            Sequence[ms.Tensor]: Classification targets for each bbox.
            Sequence[ms.Tensor]: Masks for sampled bounding boxes.
        """
        bboxes_tuple = []
        targets_tuple = []
        labels_tuple = []
        mask_tuple = []
        for i in range(self.train_batch_size):
            gt_bboxes_i = self.squeeze(gt_bboxes[i:i + 1:1, ::])
            gt_labels_i = self.squeeze(gt_labels[i:i + 1:1, ::])
            gt_labels_i = self.cast(gt_labels_i, ms.uint8)
            gt_valids_i = self.squeeze(gt_valids[i:i + 1:1, ::])
            gt_valids_i = self.cast(gt_valids_i, ms.bool_)

            bboxes, deltas, labels, mask = self.targets_generator(
                gt_bboxes_i=gt_bboxes_i,
                gt_labels_i=gt_labels_i,
                bboxes=proposal[i][::, 0:4:1],
                valid_mask=proposal_mask[i],
                gt_valids=gt_valids_i
            )

            bboxes_tuple.append(bboxes)
            labels_tuple.append(labels)
            targets_tuple.append(deltas)
            mask_tuple.append(mask)

        return bboxes_tuple, targets_tuple, labels_tuple, mask_tuple

    def get_targets_segms(
            self,
            gt_bboxes: ms.Tensor,
            gt_labels: ms.Tensor,
            gt_valids: ms.Tensor,
            proposal: Sequence[ms.Tensor],
            proposal_mask: Sequence[ms.Tensor]
    ) -> Tuple[
        Sequence[ms.Tensor], Sequence[ms.Tensor], Sequence[ms.Tensor],
        Sequence[ms.Tensor], Sequence[ms.Tensor], Sequence[ms.Tensor],
        Sequence[ms.Tensor], Sequence[ms.Tensor]
    ]:
        """Prepare proposed samples for training.

        Args:
            gt_bboxes (ms.Tensor): Ground truth bounding bboxes. Shape
                (n, self.num_gts, 4).
            gt_seg_masks (ms.Tensor): Ground truth segmentation masks.
            gt_labels (ms.Tensor): Ground truth labels for training. Shape
                (n, self.num_gts).
            gt_valids (ms.Tensor): Mask that show what GT bboxes are valid.
                Shape (n, self.num_gts).
            proposal (Sequence[ms.Tensor]): Proposed bounding bboxes. Shape
                (n, self.train_cfg.rpn_proposed.max_per_img).
            proposal_mask (Sequence[ms.Tensor]): Mask that shows what proposed
                bounding boxes are valid. Shape
                (n, self.train_cfg.rpn_proposed.max_per_img).

        Returns:
            Sequence[ms.Tensor]: Sampled bounding boxes.
            Sequence[ms.Tensor]: Localization targets for each bbox.
            Sequence[ms.Tensor]: Classification targets for each bbox.
            Sequence[ms.Tensor]: Masks for sampled bounding boxes.
            Sequence[ms.Tensor]: Sampled positive bounding boxes.
            Sequence[ms.Tensor]: Segmentation mask for each positive bbox.
            Sequence[ms.Tensor]: Classification targets for each positive bbox.
            Sequence[ms.Tensor]: Masks for sampled positive bounding boxes.
        """
        bboxes_tuple = []
        targets_tuple = []
        labels_tuple = []
        mask_tuple = []
        pos_bboxes_tuple = []
        pos_labels_tuple = []
        pos_mask_tuple = []
        pos_ids_tuple = []
        for i in range(self.train_batch_size):
            gt_bboxes_i = self.squeeze(gt_bboxes[i:i + 1:1, ::])
            gt_labels_i = self.squeeze(gt_labels[i:i + 1:1, ::])
            gt_labels_i = self.cast(gt_labels_i, ms.uint8)
            gt_valids_i = self.squeeze(gt_valids[i:i + 1:1, ::])
            gt_valids_i = self.cast(gt_valids_i, ms.bool_)
            (
                bboxes, deltas, labels, mask, pos_bboxes, pos_ids, pos_labels,
                pos_mask
            ) = self.targets_generator(
                gt_bboxes_i=gt_bboxes_i, gt_labels_i=gt_labels_i,
                valid_mask=proposal_mask[i], bboxes=proposal[i][::, 0:4:1],
                gt_valids=gt_valids_i
            )

            bboxes_tuple.append(bboxes)
            labels_tuple.append(labels)
            targets_tuple.append(deltas)
            mask_tuple.append(mask)

            pos_bboxes_tuple.append(pos_bboxes)
            pos_ids_tuple.append(pos_ids)
            pos_labels_tuple.append(pos_labels)
            pos_mask_tuple.append(pos_mask)

        return (
            bboxes_tuple, targets_tuple, labels_tuple, mask_tuple,
            pos_bboxes_tuple, pos_ids_tuple, pos_labels_tuple, pos_mask_tuple
        )


    def get_det_bboxes(
            self,
            cls_logits: ms.Tensor,
            reg_logits: ms.Tensor,
            mask_logits: ms.Tensor,
            rois: ms.Tensor,
            img_metas: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        """Get the actual detection box.

        Args:
            cls_logits (ms.Tensor): Classification predictions. Shape
                (n, self.num_classes + 1).
            reg_logits (ms.Tensor): Localization predictions. Shape
                (n, self.num_classes * 4).
            mask_logits (ms.Tensor): Mask that shows actual predictions. Shape
                (n, ).
            rois (ms.Tensor): Anchors. Shape (n, 4).
            img_metas (ms.Tensor): Information about original and final image
                sizes. Tensor. Each row contains information: orig h, orig w,
                meaning h, meaning w, padded h, padded w,
                <some additional info>.

        Returns:
            ms.Tensor: Predicted bboxes. Shape
                (self.test_batch_size, self.max_num, 5).
            ms.Tensor: Predicted labels. Shape
                (self.test_batch_size, self.max_num, 1).
            ms.Tensor: Masks that shows actual bboxes. Shape
                (self.test_batch_size, self.max_num, 1).
        """
        scores = self.softmax(cls_logits)

        boxes_all = []
        for i in range(self.num_cls_bbox):
            k = i * 4
            if self.reg_class_agnostic:
                reg_logits_i = self.squeeze(reg_logits)
            else:
                reg_logits_i = self.squeeze(reg_logits[::, k:k + 4:1])
            out_boxes_i = self.bbox_coder.decode(rois, reg_logits_i)
            boxes_all.append(out_boxes_i)

        img_metas_all = self.split(img_metas)
        scores_all = self.split(scores)
        mask_all = self.split(self.cast(mask_logits, ms.int32))

        boxes_all_with_batchsize = []
        if self.norm_bboxes:
            for i in range(self.test_batch_size):
                metas = self.split_shape(self.squeeze(img_metas_all[i])[:4])
                scale_h = metas[0] / metas[2]
                scale_w = metas[1] / metas[3]
                boxes_tuple = []
                for j in range(self.num_cls_bbox):
                    boxes_tmp = self.split(boxes_all[j])
                    out_boxes_h = ops.clip_by_value(
                        boxes_tmp[i] * scale_h,
                        clip_value_min=ms.Tensor(0, ms.float32),
                        clip_value_max=metas[0]
                    )
                    out_boxes_w = ops.clip_by_value(
                        boxes_tmp[i] * scale_w,
                        clip_value_min=ms.Tensor(0, ms.float32),
                        clip_value_max=metas[1]
                    )
                    boxes_tuple.append(
                        ops.select(self.bbox_mask, out_boxes_w, out_boxes_h),
                    )
                boxes_all_with_batchsize.append(boxes_tuple)
        else:
            for i in range(self.test_batch_size):
                boxes_tuple = []
                for j in range(self.num_cls_bbox):
                    boxes_tmp = self.split(boxes_all[j])
                    boxes_tuple.append(boxes_tmp[i])
                boxes_all_with_batchsize.append(boxes_tuple)

        res_bboxes_nms, res_labels_nms, res_mask_nms = self._multiclass_nms(
            boxes_all=boxes_all_with_batchsize, scores_all=scores_all,
            mask_all=mask_all
        )

        res_bboxes = []
        res_labels = []
        res_mask = []
        for i in range(self.test_batch_size):
            res_bboxes_, res_labels_, res_mask_ = self._get_best(
                res_bboxes_nms[i], res_labels_nms[i], res_mask_nms[i]
            )
            res_bboxes.append(res_bboxes_)
            res_labels.append(res_labels_)
            res_mask.append(res_mask_)

        res_bboxes = ops.concat(res_bboxes).reshape((-1, self.max_num, 5))
        res_labels = ops.concat(res_labels).reshape((-1, self.max_num, 1))
        res_mask = ops.concat(res_mask).reshape((-1, self.max_num, 1))
        return res_bboxes, res_labels, res_mask

    def _get_best(
            self, bboxes: ms.Tensor, labels: ms.Tensor, masks: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        """Filter predicted bboxes by score."""
        score = bboxes[::, 4] * masks.reshape(-1)
        _, score_indicies = self.test_topk(score, self.max_num)

        bboxes = bboxes[score_indicies]
        labels = labels[score_indicies]
        masks = masks[score_indicies]

        return bboxes, labels, masks

    def _multiclass_nms(
            self,
            boxes_all: Sequence[Sequence[ms.Tensor]],
            scores_all: Sequence[ms.Tensor],
            mask_all: Sequence[ms.Tensor]
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        """Bounding boxes postprocessing."""
        all_bboxes = []
        all_labels = []
        all_masks = []

        for i in range(self.test_batch_size):
            bboxes = boxes_all[i]
            scores = scores_all[i]
            masks = ops.cast(mask_all[i], ms.bool_)

            res_boxes_tuple = []
            res_labels_tuple = []
            res_masks_tuple = []
            for j in range(self.num_classes):
                k = j + 1
                _cls_scores = scores[::, k:k + 1:1]
                _bboxes = ops.squeeze(bboxes[j])
                _mask_o = ops.reshape(masks, (self.rpn_max_num, 1))

                cls_mask = ops.gt(_cls_scores, self.test_score_thresh)
                _mask = ops.logical_and(_mask_o, cls_mask)

                _reg_mask = ops.cast(
                    ops.tile(ops.cast(_mask, ms.int32), (1, 4)), ms.bool_
                )

                _bboxes = ops.select(_reg_mask, _bboxes, self.test_box_zeros)
                _cls_scores = ops.select(
                    _mask, _cls_scores, self.test_score_zeros
                )
                __cls_scores = ops.squeeze(_cls_scores)
                scores_sorted, topk_inds = self.test_topk(
                    __cls_scores, self.rpn_max_num
                )
                topk_inds = ops.reshape(topk_inds, (self.rpn_max_num, 1))
                scores_sorted = ops.reshape(
                    scores_sorted, (self.rpn_max_num, 1)
                )
                _bboxes_sorted = ops.gather_nd(_bboxes, topk_inds)
                _mask_sorted = ops.gather_nd(_mask, topk_inds)

                scores_sorted = ops.tile(scores_sorted, (1, 4))
                cls_dets = ops.concat(
                    (_bboxes_sorted, scores_sorted), axis=1
                )
                cls_dets = ops.slice(cls_dets, (0, 0), (self.rpn_max_num, 5))

                cls_dets, _index, _mask_nms = self.nms_test(cls_dets)
                cls_dets, _index, _mask_nms = [
                    ops.stop_gradient(a)
                    for a in (cls_dets, _index, _mask_nms)
                ]

                _index = ops.reshape(_index, (self.rpn_max_num, 1))
                _mask_nms = ops.reshape(_mask_nms, (self.rpn_max_num, 1))

                _mask_n = ops.gather_nd(_mask_sorted, _index)
                _mask_n = ops.logical_and(_mask_n, _mask_nms)
                cls_labels = ops.ones_like(_index) * j
                res_boxes_tuple.append(cls_dets)
                res_labels_tuple.append(cls_labels)
                res_masks_tuple.append(_mask_n)

            res_boxes = ops.concat(res_boxes_tuple, axis=0)
            res_labels = ops.concat(res_labels_tuple, axis=0)
            res_masks = ops.concat(res_masks_tuple, axis=0)

            reshape_size = self.num_classes * self.rpn_max_num
            res_boxes = ops.reshape(res_boxes, (1, reshape_size, 5))
            res_labels = ops.reshape(res_labels, (1, reshape_size, 1))
            res_masks = ops.reshape(res_masks, (1, reshape_size, 1))

            all_bboxes.append(res_boxes)
            all_labels.append(res_labels)
            all_masks.append(res_masks)

        all_bboxes = ops.concat(all_bboxes, axis=0)
        all_labels = ops.concat(all_labels, axis=0)
        all_masks = ops.concat(all_masks, axis=0)
        return all_bboxes, all_labels, all_masks
