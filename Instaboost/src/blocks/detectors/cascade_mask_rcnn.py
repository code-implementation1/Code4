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
"""Cascade RCNN"""
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


class CascadeRCNN(nn.Cell):
    """CascadeRCNN for instance segmentation task."""

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

        self.bbox_head_0 = self.create_bbox_head(config, stage=0)
        self.bbox_head_1 = self.create_bbox_head(config, stage=1)
        self.bbox_head_2 = self.create_bbox_head(config, stage=2)
        self.mask_head_0 = self.create_mask_head(config)
        self.mask_head_1 = self.create_mask_head(config)
        self.mask_head_2 = self.create_mask_head(config)

        bbox_head_cfg_0 = config.bbox_head.as_dict().copy()
        bbox_coder_cfg_0 = bbox_head_cfg_0.pop('bbox_coder').as_dict().copy()
        self.decode_0 = DeltaXYWHBBoxCoder(**bbox_coder_cfg_0)

        bbox_head_cfg_1 = config.bbox_head_1.as_dict().copy()
        bbox_coder_cfg_1 = bbox_head_cfg_1.pop('bbox_coder').as_dict().copy()
        self.decode_1 = DeltaXYWHBBoxCoder(**bbox_coder_cfg_1)
        self.max_shape = (config.img_height, config.img_width)

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
        loss_bbox = nn.SmoothL1Loss(reduction='none', **loss_bbox_cfg)

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

    def create_bbox_head(self, config, stage=None):
        if stage == 0:
            bbox_head_cfg = config.bbox_head.as_dict().copy()
        elif stage == 1:
            bbox_head_cfg = config.bbox_head_1.as_dict().copy()
        elif stage == 2:
            bbox_head_cfg = config.bbox_head_2.as_dict().copy()
        else:
            raise NotImplementedError('Not implemented for stages more than 3')
        bbox_coder_cfg = bbox_head_cfg.pop('bbox_coder').as_dict().copy()
        loss_cls_cfg = bbox_head_cfg.pop('loss_cls').as_dict().copy()
        loss_bbox_cfg = bbox_head_cfg.pop('loss_bbox').as_dict().copy()

        loss_cls_weight = loss_cls_cfg.pop('loss_weight', 1.0)
        loss_bbox_weight = loss_bbox_cfg.pop('loss_weight', 1.0)

        loss_cls = nn.SoftmaxCrossEntropyWithLogits(
            reduction='none', **loss_cls_cfg
        )
        loss_bbox = nn.L1Loss(reduction='none')

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
        """Construct the CascadeRCNN Network.

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
            output = self.run_train()
        else:
            output = self.run_test(x, img_metas)

        return output

    def run_train(self):
        """Run CascadeRCNN loss calculation."""
        raise NotImplementedError('Cascade mask rcnn training is not implemented.')

    def run_test(self, feats, img_metas):
        """Run prediction calculation."""
        cls_score, bbox_pred = self.rpn_head(feats)
        proposal, proposal_mask = self.rpn_head.get_bboxes(
            cls_score, bbox_pred, img_metas)
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
            [ops.cast(feat, ms.float32)
             for feat in feats[:self.roi_align_test.num_levels]]
        )
        roi_feats = ops.cast(roi_feats, self.ms_type)
        rcnn_masks = ops.concat(mask_tuple, axis=0)
        rcnn_masks = ops.stop_gradient(rcnn_masks)
        cls_logits, reg_logits = self.bbox_head_0(roi_feats)

        rois_1 = self.get_rois_decode(rois, reg_logits, stage=0)
        rois_1 = ops.squeeze(rois_1)
        rois_1 = ops.concat(
            (self.roi_align_index_test_tensor, rois_1), axis=1
        )
        rois_1 = ops.cast(rois_1, ms.float32)
        rois_1 = ops.stop_gradient(rois_1)
        roi_feats_1 = self.roi_align_test(
            rois_1,
            [ops.cast(feat, ms.float32)
             for feat in feats[:self.roi_align_test.num_levels]]
        )
        roi_feats_1 = ops.cast(roi_feats_1, self.ms_type)
        cls_logits_1, reg_logits_1 = self.bbox_head_1(roi_feats_1)

        rois_2 = self.get_rois_decode(rois_1, reg_logits_1, stage=1)
        bboxes_all_2 = ops.reshape(rois_2, (-1, 4))
        rois_2 = ops.squeeze(rois_2)
        rois_2 = ops.concat(
            (self.roi_align_index_test_tensor, rois_2), axis=1
        )
        rois_2 = ops.cast(rois_2, ms.float32)
        rois_2 = ops.stop_gradient(rois_2)
        roi_feats_2 = self.roi_align_test(
            rois_2,
            [ops.cast(feat, ms.float32)
             for feat in feats[:self.roi_align_test.num_levels]]
        )
        roi_feats_2 = ops.cast(roi_feats_2, self.ms_type)
        cls_logits_2, reg_logits_2 = self.bbox_head_2(roi_feats_2)

        cls_logits_mean = ops.cat((
            ops.unsqueeze(cls_logits, 0),
            ops.unsqueeze(cls_logits_1, 0),
            ops.unsqueeze(cls_logits_2, 0),
        ), 0).mean(0)
        res_bboxes, res_labels, res_mask = self.bbox_head_2.get_det_bboxes(
            cls_logits_mean, reg_logits_2, rcnn_masks, bboxes_all_2, img_metas)
        res_bboxes_reshaped = res_bboxes.reshape((-1, 5))
        mask_rois = ops.concat(
            (
                self.mask_roi_align_index_test_tensor,
                res_bboxes_reshaped[..., :4]
            ), axis=1)
        mask_rois = self.cast(mask_rois, ms.float32)
        mask_rois = ops.stop_gradient(mask_rois)
        mask_roi_feats = self.mask_roi_align_test(
            mask_rois,
            [ops.cast(feat, ms.float32)
             for feat in feats[:self.roi_align_test.num_levels]]
        )
        mask_roi_feats = self.cast(mask_roi_feats, self.ms_type)
        mask_logits = self.mask_head_0(mask_roi_feats)
        mask_logits_1 = self.mask_head_1(mask_roi_feats)
        mask_logits_2 = self.mask_head_2(mask_roi_feats)
        mask_logits_mean = ops.cat((ops.unsqueeze(mask_logits, 0),
                                    ops.unsqueeze(mask_logits_1, 0),
                                    ops.unsqueeze(mask_logits_2, 0),
                                    ), 0).mean(0)
        res_pred_mask = self.mask_head_2.choose_masks(
            mask_logits=mask_logits_mean, det_labels=res_labels
        )
        return (
            res_bboxes, res_labels, res_pred_mask, res_mask, None, None, None
        )

    def get_rois_decode(self, rois, bbox_pred, stage=None, batch_size=1):
        boxes = rois[:, 1:5]
        boxes = self.cast(boxes, ms.float32)
        box_deltas = bbox_pred
        if stage == 0:
            pred_boxes = self.decode_0.decode(boxes,
                                              box_deltas,
                                              self.max_shape)
        else:
            pred_boxes = self.decode_1.decode(boxes,
                                              box_deltas,
                                              self.max_shape)
        ret_boxes = pred_boxes.view(batch_size, -1, 4)
        return ret_boxes

    def set_train(self, mode=True):
        """Change training mode."""
        super().set_train(mode=mode)
        self.backbone.set_train(mode=mode)


class CascadeRCNNInfer(nn.Cell):
    """CascadeRCNN wrapper for inference."""

    def __init__(self, config):
        super().__init__()
        self.net = CascadeRCNN(config)
        self.net.set_train(False)

    def construct(self, img_data, img_metas=None):
        """Make predictions."""
        output = self.net(img_data, img_metas, None, None, None, None)
        bboxes, labels, mask_pred, valids, _, _, _ = output
        return bboxes, labels, mask_pred, valids
