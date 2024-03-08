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
"""Mask RCNN"""
import numpy as np
import mindspore as ms
from mindspore import ops, nn
from mindspore.common.tensor import Tensor

from ..dense_heads import RPNHead
from ..roi_extractors import SingleRoIExtractor
from ..bbox_heads import Shared2FCBBoxHead
from ..backbones import ResNet, ResNeXt
from ..necks import FPN
from ..bbox_coders import DeltaXYWHBBoxCoder
from ..assigners_samplers import (
    build_mask_max_iou_random, build_max_iou_random
)
from ..anchor_generator import AnchorGenerator
from ..mask_heads import FCNMaskHead


class MaskRCNN(nn.Cell):
    """MaskRCNN for instance segmentation task."""

    def __init__(self, config):
        super().__init__()
        self.dtype = np.float32
        self.ms_type = ms.float32
        self.train_batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size
        self.num_classes = config.num_classes
        self.max_num = config.num_gts

        # Backbone
        self.backbone = self.create_backbone(config)

        self.neck = FPN(
            in_channels=config.neck.fpn.in_channels,
            out_channels=config.neck.fpn.out_channels,
            num_outs=config.neck.fpn.num_outs,
            feature_shapes=config.feature_shapes
        )

        # RPN and rpn loss
        self.gt_labels_stage1 = Tensor(
            np.ones((self.train_batch_size, config.num_gts)).astype(np.uint8)
        )
        self.rpn_head = self.create_rpn_head(config)

        # RoI
        self.roi_align, self.roi_align_test = self.create_roi(
            roi_layer=config.roi.roi_layer,
            roi_align_featmap_strides=config.roi.featmap_strides,
            roi_align_out_channels=config.roi.out_channels,
            roi_align_finest_scale=config.roi.finest_scale,
            train_batch_size=config.roi.sample_num * self.train_batch_size,
            test_batch_size=config.test_cfg.rpn.max_per_img * self.test_batch_size
        )

        self.mask_roi_align, self.mask_roi_align_test = self.create_roi(
            roi_layer=config.mask_roi.roi_layer,
            roi_align_featmap_strides=config.mask_roi.featmap_strides,
            roi_align_out_channels=config.mask_roi.out_channels,
            roi_align_finest_scale=config.mask_roi.finest_scale,
            train_batch_size=config.mask_roi.sample_num * self.train_batch_size,
            test_batch_size=config.test_cfg.rcnn.max_per_img * self.test_batch_size
        )

        # Init tensor
        (
            self.roi_align_index_tensor,
            self.roi_align_index_test_tensor,
            self.mask_roi_align_index_tensor,
            self.mask_roi_align_index_test_tensor
        ) = self.create_roi_tensors(config)

        self.bbox_head = self.create_bbox_head(config)
        self.mask_head = self.create_mask_head(config)
        self.mask_divider = config.mask_divider

    @staticmethod
    def create_backbone(config):
        """Create backbone and init it."""
        if config.backbone.type == 'resnet':
            backbone = ResNet(
                depth=config.backbone.depth,
                num_stages=config.backbone.num_stages,
                strides=(1, 2, 2, 2),
                dilations=(1, 1, 1, 1),
                out_indices=config.backbone.out_indices,
                frozen_stages=config.backbone.frozen_stages,
                norm_eval=config.backbone.norm_eval
            )
        elif config.backbone.type == 'resnext':
            backbone = ResNeXt(
                depth=config.backbone.depth,
                num_stages=config.backbone.num_stages,
                strides=(1, 2, 2, 2),
                dilations=(1, 1, 1, 1),
                out_indices=config.backbone.out_indices,
                frozen_stages=config.backbone.frozen_stages,
                norm_eval=config.backbone.norm_eval,
                groups=config.backbone.groups,
                base_width=config.backbone.base_width
            )
        else:
            raise ValueError(f'Unsupported backbone: {config.backbone.type}')

        return backbone

    def create_rpn_head(self, config):
        rpn_cfg = config.rpn.as_dict().copy()
        bbox_coder_cfg = rpn_cfg.pop('bbox_coder').as_dict().copy()
        loss_cls_cfg = rpn_cfg.pop('loss_cls').as_dict().copy()
        loss_bbox_cfg = rpn_cfg.pop('loss_bbox').as_dict().copy()
        anchor_generator_cfg = rpn_cfg.pop('anchor_generator').as_dict().copy()

        loss_cls_weight = loss_cls_cfg.pop('loss_weight', 1.0)
        loss_bbox_weight = loss_bbox_cfg.pop('loss_weight', 1.0)

        loss_cls = ops.SigmoidCrossEntropyWithLogits(**loss_cls_cfg)
        loss_bbox = nn.L1Loss(reduction='none')

        bbox_coder = DeltaXYWHBBoxCoder(**bbox_coder_cfg)

        anchor_generator = AnchorGenerator(**anchor_generator_cfg)
        num_bboxes = sum(
            anchor_generator.num_base_anchors[i] * sh[0] * sh[1]
            for i, sh in enumerate(config.feature_shapes)
        )

        assigner_sampler = build_max_iou_random(
            num_bboxes=num_bboxes,
            num_gts=config.num_gts,
            assigner_cfg=config.train_cfg.rpn.assigner.as_dict().copy(),
            sampler_cfg=config.train_cfg.rpn.sampler.as_dict().copy(),
            bbox_coder=bbox_coder,
            rcnn_mode=False
        )

        return RPNHead(
            feature_shapes=config.feature_shapes,
            train_batch_size=config.batch_size,
            test_batch_size=config.test_batch_size,
            num_gts=config.num_gts,
            **rpn_cfg,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            targets_generator=assigner_sampler,
            loss_cls=loss_cls,
            loss_cls_weight=loss_cls_weight,
            loss_bbox=loss_bbox,
            loss_bbox_weight=loss_bbox_weight,
            train_cfg=config.train_cfg,
            test_cfg=config.test_cfg
        )

    def create_roi(
            self, roi_layer, roi_align_out_channels, roi_align_featmap_strides,
            roi_align_finest_scale, train_batch_size, test_batch_size
    ):
        """Create roi extraction blocks in training and inference mode."""
        roi_align = SingleRoIExtractor(
            roi_layer=roi_layer,
            out_channels=roi_align_out_channels,
            featmap_strides=roi_align_featmap_strides,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            finest_scale=roi_align_finest_scale
        )
        roi_align.set_train_local(True)
        roi_align_test = SingleRoIExtractor(
            roi_layer=roi_layer,
            out_channels=roi_align_out_channels,
            featmap_strides=roi_align_featmap_strides,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            finest_scale=roi_align_finest_scale
        )
        roi_align_test.set_train_local(False)

        return roi_align, roi_align_test

    def create_bbox_head(self, config):
        bbox_head_cfg = config.bbox_head.as_dict().copy()
        bbox_coder_cfg = bbox_head_cfg.pop('bbox_coder').as_dict().copy()
        loss_cls_cfg = bbox_head_cfg.pop('loss_cls').as_dict().copy()
        loss_bbox_cfg = bbox_head_cfg.pop('loss_bbox').as_dict().copy()

        loss_cls_weight = loss_cls_cfg.pop('loss_weight', 1.0)
        loss_bbox_weight = loss_bbox_cfg.pop('loss_weight', 1.0)

        loss_cls = nn.SoftmaxCrossEntropyWithLogits(
            reduction='none', **loss_cls_cfg
        )
        loss_bbox = nn.L1Loss(reduction='none', **loss_bbox_cfg)

        bbox_coder = DeltaXYWHBBoxCoder(**bbox_coder_cfg)

        assigner_sampler = build_mask_max_iou_random(
            num_bboxes=config.train_cfg.rpn_proposal.max_per_img,
            num_gts=config.num_gts,
            assigner_cfg=config.train_cfg.rcnn.assigner.as_dict().copy(),
            sampler_cfg=config.train_cfg.rcnn.sampler.as_dict().copy(),
            bbox_coder=bbox_coder,
            rcnn_mode=True
        )

        return Shared2FCBBoxHead(
            train_batch_size=config.batch_size,
            test_batch_size=config.test_batch_size,
            num_gts=config.num_gts,
            **bbox_head_cfg,
            num_classes=config.num_classes - 1,
            bbox_coder=bbox_coder,
            targets_generator=assigner_sampler,
            loss_cls=loss_cls,
            loss_cls_weight=loss_cls_weight,
            loss_bbox=loss_bbox,
            loss_bbox_weight=loss_bbox_weight,
            test_cfg=config.test_cfg
        )

    def create_mask_head(self, config):
        mask_head_cfg = config.mask_head.as_dict().copy()
        loss_mask_cfg = mask_head_cfg.pop('loss_mask').as_dict().copy()
        loss_mask_weight = loss_mask_cfg.pop('loss_weight', 1.0)

        loss_mask = ops.SigmoidCrossEntropyWithLogits(**loss_mask_cfg)
        return FCNMaskHead(
            **mask_head_cfg,
            num_classes=config.num_classes - 1,
            loss_mask=loss_mask,
            train_cfg=config.train_cfg,
            test_cfg=config.test_cfg,
            test_batch_size=config.test_batch_size,
            train_batch_size=config.batch_size,
            loss_mask_weight=loss_mask_weight
        )

    def create_roi_tensors(self, config):
        """Init some helpful tensors."""
        num_expected_total = int(
            config.train_cfg.rcnn.sampler.num *
            (config.train_cfg.rcnn.sampler.pos_fraction + 1.)
        )
        num_expected_pos = int(
            config.train_cfg.rcnn.sampler.num *
            config.train_cfg.rcnn.sampler.pos_fraction
        )
        roi_align_index = [
            np.array(
                np.ones((num_expected_total, 1)) * i, dtype=self.dtype
            ) for i in range(self.train_batch_size)
        ]
        mask_roi_align_index = [
            np.array(
                np.ones((num_expected_pos, 1)) * i, dtype=self.dtype
            ) for i in range(self.train_batch_size)
        ]
        roi_align_index_test = [
            np.array(
                np.ones((config.test_cfg.rpn.max_per_img, 1)) * i,
                dtype=self.dtype
            )
            for i in range(self.test_batch_size)
        ]
        mask_roi_align_index_test = [
            np.array(
                np.ones((config.test_cfg.rcnn.max_per_img, 1)) * i,
                dtype=self.dtype
            )
            for i in range(self.test_batch_size)
        ]

        roi_align_index_tensor = Tensor(np.concatenate(roi_align_index))
        mask_roi_align_index_tensor = Tensor(
            np.concatenate(mask_roi_align_index)
        )
        roi_align_index_test_tensor = Tensor(
            np.concatenate(roi_align_index_test)
        )
        mask_roi_align_index_test_tensor = Tensor(
            np.concatenate(mask_roi_align_index_test)
        )

        return (
            roi_align_index_tensor, roi_align_index_test_tensor,
            mask_roi_align_index_tensor, mask_roi_align_index_test_tensor
        )

    def construct(
            self, img_data, img_metas, gt_bboxes, gt_seg_masks, gt_labels,
            gt_valids
    ):
        """Construct the MaskRCNN Network.

        Args:
            img_data (Tensor):
                Input image data.
            img_metas (Tensor):
                Meta label of img.
            gt_bboxes (Tensor):
                Get the value of bboxes.
            gt_seg_masks (Tensor):
                Get the value of masks.
            gt_labels (Tensor):
                Get the value of labels.
            gt_valids (Tensor):
                Get the valid part of bboxes.

        Returns:
            Tuple,tuple of output tensor (losses if training else predictions).
        """
        x = self.backbone(img_data)
        x = self.neck(x)
        if self.training:
            output = self.run_train(
                x, img_metas, gt_bboxes, gt_seg_masks, gt_labels, gt_valids
            )
        else:
            output = self.run_test(x, img_metas)

        return output

    def run_train(
            self, feats, img_metas, gt_bboxes, gt_seg_masks, gt_labels,
            gt_valids
    ):
        """Run MaskRCNN loss calculation."""
        # Compute RPN loss.
        cls_score, bbox_pred = self.rpn_head(feats)
        (
            rpn_bbox_targets, rpn_bbox_weights, rpn_labels, rpn_label_weights
        ) = self.rpn_head.get_targets(
            gt_bboxes, self.gt_labels_stage1, gt_valids, img_metas
        )
        rpn_loss, rpn_cls_loss, rpn_reg_loss = self.rpn_head.loss(
            rpn_bbox_targets, rpn_bbox_weights, rpn_labels, rpn_label_weights,
            cls_score, bbox_pred
        )

        # Generate proposals
        proposal, proposal_mask = self.rpn_head.get_bboxes(
            cls_score, bbox_pred, img_metas
        )
        proposal = [ops.stop_gradient(p) for p in proposal]
        proposal_mask = [ops.stop_gradient(p) for p in proposal_mask]

        gt_labels = ops.cast(gt_labels, ms.int32)
        gt_valids = ops.cast(gt_valids, ms.int32)

        # Generate targets for rcnn and extract RoIs
        (
            bboxes_tuple, deltas_tuple, labels_tuple, valid_tuple,
            pos_bboxes_tuple, seg_tuple, pos_labels_tuple, pos_valids_tuple
        ) = self.bbox_head.get_targets_segms(
            gt_bboxes=gt_bboxes, gt_seg_masks=gt_seg_masks,
            gt_labels=gt_labels, gt_valids=gt_valids, proposal=proposal,
            proposal_mask=proposal_mask
        )

        bbox_targets = ops.concat(deltas_tuple, axis=0)
        bbox_targets = ops.stop_gradient(bbox_targets)

        rcnn_labels = ops.concat(labels_tuple, axis=0)
        rcnn_labels = ops.cast(rcnn_labels, ms.int32)
        rcnn_labels = ops.stop_gradient(rcnn_labels)

        rcnn_valid = ops.concat(valid_tuple, axis=0)
        rcnn_valid = ops.stop_gradient(rcnn_valid)
        rcnn_valid_squeeze = ops.squeeze(ops.cast(rcnn_valid, ms.bool_))

        bboxes_all = ops.concat(bboxes_tuple, axis=0)

        rois = ops.concat((self.roi_align_index_tensor, bboxes_all), axis=1)
        rois = ops.cast(rois, ms.float32)
        rois = ops.stop_gradient(rois)

        roi_feats = self.roi_align(
            rois,
            [
                ops.cast(feat, ms.float32)
                for feat in feats[:self.roi_align.num_levels]
            ]
        )

        roi_feats = ops.cast(roi_feats, self.ms_type)

        # Compute RCNN loss
        cls_logits, reg_logits = self.bbox_head(roi_feats)
        rcnn_loss, rcnn_cls_loss, rcnn_reg_loss = self.bbox_head.loss(
            cls_score=cls_logits,
            bbox_pred=reg_logits,
            bbox_targets=bbox_targets,
            labels=rcnn_labels,
            weights=rcnn_valid_squeeze
        )

        # Compute segmentation loss
        seg_labels = ops.concat(pos_labels_tuple, axis=0)
        seg_labels = ops.stop_gradient(seg_labels)

        seg_targets = ops.concat(seg_tuple, axis=0)
        seg_targets = ops.stop_gradient(seg_targets)

        seg_bboxes = ops.concat(pos_bboxes_tuple, axis=0)
        seg_bboxes = ops.stop_gradient(seg_bboxes)

        seg_masks_valids = ops.concat(pos_valids_tuple, axis=0)
        seg_masks_valids = ops.squeeze(ops.cast(seg_masks_valids, ms.bool_))
        seg_masks_valids = ops.stop_gradient(seg_masks_valids)

        seg_rois = ops.concat(
            (self.mask_roi_align_index_tensor, seg_bboxes), axis=1
        )
        seg_rois = ops.cast(seg_rois, ms.float32)
        seg_rois = ops.stop_gradient(seg_rois)

        mask_roi_feats = self.mask_roi_align(
            seg_rois,
            [
                ops.cast(feat, ms.float32)
                for feat in feats[:self.roi_align.num_levels]
            ]
        )

        seg_logits = self.mask_head(mask_roi_feats)

        seg_cropped_targets = self.mask_head.get_targets(
            seg_bboxes, seg_targets, self.mask_divider
        )

        seg_cropped_targets = ops.stop_gradient(seg_cropped_targets)
        # seg_logits, labels, weights, seg_targets
        seg_loss = self.mask_head.loss(
            seg_logits=seg_logits, labels=seg_labels, weights=seg_masks_valids,
            seg_targets=seg_cropped_targets
        )

        return (
            rpn_loss, rcnn_loss, seg_loss, rpn_cls_loss, rpn_reg_loss,
            rcnn_cls_loss, rcnn_reg_loss
        )

    def run_test(self, feats, img_metas):
        """Run prediction calculation."""
        cls_score, bbox_pred = self.rpn_head(feats)
        proposal, proposal_mask = self.rpn_head.get_bboxes(
            cls_score, bbox_pred, img_metas
        )
        bboxes_tuple = ()
        mask_tuple = ()

        mask_tuple += proposal_mask
        for p_i in proposal:
            bboxes_tuple += (p_i[::, 0:4:1],)

        if self.test_batch_size > 1:
            bboxes_all = ops.concat(bboxes_tuple, axis=0)
        else:
            bboxes_all = bboxes_tuple[0]

        rois = ops.concat(
            (self.roi_align_index_test_tensor, bboxes_all), axis=1
        )
        rois = ops.cast(rois, ms.float32)
        rois = ops.stop_gradient(rois)
        roi_feats = self.roi_align_test(
            rois,
            [
                ops.cast(feat, ms.float32)
                for feat in feats[:self.roi_align_test.num_levels]
            ]
        )

        roi_feats = ops.cast(roi_feats, self.ms_type)
        rcnn_masks = ops.concat(mask_tuple, axis=0)
        rcnn_masks = ops.stop_gradient(rcnn_masks)

        cls_logits, reg_logits = self.bbox_head(roi_feats)

        res_bboxes, res_labels, res_mask = self.bbox_head.get_det_bboxes(
            cls_logits, reg_logits, rcnn_masks, bboxes_all, img_metas
        )

        res_bboxes_reshaped = res_bboxes.reshape((-1, 5))
        mask_rois = ops.concat(
            (
                self.mask_roi_align_index_test_tensor,
                res_bboxes_reshaped[..., :4]
            ),
            axis=1
        )
        mask_rois = self.cast(mask_rois, ms.float32)
        mask_rois = ops.stop_gradient(mask_rois)
        mask_roi_feats = self.mask_roi_align_test(
            mask_rois,
            [
                ops.cast(feat, ms.float32)
                for feat in feats[:self.roi_align_test.num_levels]
            ]
        )
        mask_roi_feats = self.cast(mask_roi_feats, self.ms_type)
        mask_logits = self.mask_head(mask_roi_feats)

        res_pred_mask = self.mask_head.choose_masks(
            mask_logits=mask_logits, det_labels=res_labels
        )

        return (
            res_bboxes, res_labels, res_pred_mask, res_mask, None, None, None
        )

    def set_train(self, mode=True):
        """Change training mode."""
        super().set_train(mode=mode)
        self.backbone.set_train(mode=mode)


class MaskRCNNInfer(nn.Cell):
    """MaskRCNN wrapper for inference."""

    def __init__(self, config):
        super().__init__()
        self.net = MaskRCNN(config)
        self.net.set_train(False)

    def construct(self, img_data, img_metas=None):
        """Make predictions."""
        output = self.net(img_data, img_metas, None, None, None, None)
        bboxes, labels, mask_pred, valids, _, _, _ = output
        return bboxes, labels, mask_pred, valids
