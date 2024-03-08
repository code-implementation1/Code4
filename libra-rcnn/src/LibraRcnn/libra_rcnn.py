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
"""LibraRcnn"""
import logging
import os

import numpy as np
import mindspore as ms
from mindspore import ops, nn
from mindspore.common.tensor import Tensor

from .proposal_generator import Proposal
from .rpn import RPN
from .roi_align import SingleRoIExtractor
from .anchor_generator import AnchorGenerator
from .bbox_head_libra import Shared2FCBBoxHead
from .resnet import ResNet
from .resnext import ResNeXt
from .neck.fpn import FPN
from .neck.bfp import BFP


class LibraRcnn(nn.Cell):
    """
    LibraRcnn Network.

    Examples:
        net = LibraRcnn(config)
    """

    def __init__(self, config):
        super().__init__()
        self.dtype = np.float32
        self.ms_type = ms.float32
        self.train_batch_size = config.batch_size
        self.test_batch_size = config.test_batch_size
        self.num_classes = config.num_classes
        self.max_num = config.num_gts

        # Anchor generator
        self.anchor_generator, self.anchor_list = self.create_anchors(config)

        # Backbone
        self.backbone = self.create_backbone(config)
        if (config.backbone.pretrained is not None
                and os.path.exists(config.backbone.pretrained)):
            logging.info('Load backbone weights...')
            ms.load_checkpoint(config.backbone.pretrained, self.backbone)
        else:
            logging.info('Backbone weights were not loaded.')

        # Neck
        self.neck = nn.SequentialCell(
            FPN(
                in_channels=config.neck.fpn.in_channels,
                out_channels=config.neck.fpn.out_channels,
                num_outs=config.neck.fpn.num_outs,
                feature_shapes=config.feature_shapes
            ),
            BFP(
                in_channels=config.neck.bfp.in_channels,
                num_levels=config.neck.bfp.num_levels,
                refine_level=config.neck.bfp.refine_level,
                refine_type=config.neck.bfp.refine_type,
                feature_shapes=config.feature_shapes,
            )
        )

        # Rpn and rpn loss
        self.gt_labels_stage1 = Tensor(
            np.ones((self.train_batch_size, config.num_gts)).astype(np.uint8)
        )
        self.rpn_head = RPN(
            num_bboxes=config.num_bboxes, feature_shapes=config.feature_shapes,
            batch_size=config.batch_size, in_channels=config.rpn.in_channels,
            feat_channels=config.rpn.feat_channels,
            cls_out_channels=config.rpn.cls_out_channels,
            num_anchors=self.anchor_generator.num_base_anchors[0],
            cls_loss=config.rpn.cls_loss, reg_loss=config.rpn.reg_loss,
            bbox_assign_sampler=config.rpn.bbox_assign_sampler,
            num_gts=config.num_gts,
            target_means=config.rpn.target_means,
            target_stds=config.rpn.target_stds,
        )

        # Proposal
        (
            self.proposal_generator, self.proposal_generator_test
        ) = self.create_proposal_generator(config)

        # Roi
        self.roi_align, self.roi_align_test = self.create_roi(
            roi_layer=config.roi.roi_layer,
            roi_align_featmap_strides=config.roi.featmap_strides,
            roi_align_out_channels=config.roi.out_channels,
            roi_align_finest_scale=config.roi.finest_scale,
            train_batch_size=config.roi.sample_num * self.train_batch_size,
            test_batch_size=config.proposal.test.max_num * self.test_batch_size
        )

        # Init tensor
        (
            self.roi_align_index_tensor, self.roi_align_index_test_tensor
        ) = self.create_roi_tensor(config)

        self.bbox_head = Shared2FCBBoxHead(
            in_channels=config.rcnn.in_channels,
            fc_out_channels=config.rcnn.fc_out_channels,
            roi_feat_size=config.rcnn.roi_feat_size,
            num_classes=self.num_classes - 1,
            loss_cls=config.rcnn.cls_loss,
            loss_bbox=config.rcnn.reg_loss,
            bbox_coder=config.rcnn.bbox_coder,
            train_batch_size=self.train_batch_size,
            test_batch_size=self.test_batch_size,
            num_gts=config.num_gts,
            assign_sampler_config=config.rcnn.assign_sampler,
            test_bboxes_num=config.proposal.test.max_num,
            score_thr=config.rcnn.score_thr,
            iou_thr=config.rcnn.iou_thr
        )

    def create_anchors(self, config):
        """Init anchor generators and anchors."""
        anchor_generator = AnchorGenerator(
            strides=config.anchor_generator.strides,
            ratios=config.anchor_generator.ratios,
            scales=config.anchor_generator.scales
        )
        featmap_sizes = config.feature_shapes
        anchor_list = anchor_generator.grid_priors(featmap_sizes)
        return anchor_generator, anchor_list

    def create_backbone(self, config):
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


    def create_proposal_generator(self, config):
        """Create proposal generator."""
        proposal_generator = Proposal(
            batch_size=config.batch_size,
            num_classes=config.proposal.activate_num_classes,
            use_sigmoid_cls=config.proposal.use_sigmoid_cls,
            feature_shapes=config.feature_shapes,
            img_height=config.img_height,
            img_width=config.img_width,
            nms_pre=config.proposal.train.nms_pre,
            nms_thr=config.proposal.train.nms_thr,
            max_num=config.proposal.train.max_num,
            num_levels=config.neck.fpn.num_outs,
            target_means=config.rpn.target_means,
            target_stds=config.rpn.target_stds
        )
        proposal_generator_test = Proposal(
            batch_size=config.test_batch_size,
            num_classes=config.proposal.activate_num_classes,
            use_sigmoid_cls=config.proposal.use_sigmoid_cls,
            feature_shapes=config.feature_shapes,
            img_height=config.img_height,
            img_width=config.img_width,
            nms_pre=config.proposal.test.nms_pre,
            nms_thr=config.proposal.test.nms_thr,
            max_num=config.proposal.test.max_num,
            num_levels=config.neck.fpn.num_outs,
            target_means=config.rpn.target_means,
            target_stds=config.rpn.target_stds
        )

        return proposal_generator, proposal_generator_test

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

    def create_roi_tensor(self, config):
        """Init some helpful tensors."""
        num_expected_total = (
            config.rcnn.assign_sampler.num_expected_pos +
            config.rcnn.assign_sampler.num_expected_neg
        )
        roi_align_index = [
            np.array(
                np.ones((num_expected_total, 1)) * i, dtype=self.dtype
            ) for i in range(self.train_batch_size)
        ]
        roi_align_index_test = [
            np.array(
                np.ones((config.proposal.test.max_num, 1)) * i,
                dtype=self.dtype
            )
            for i in range(self.test_batch_size)
        ]

        roi_align_index_tensor = Tensor(np.concatenate(roi_align_index))
        roi_align_index_test_tensor = Tensor(
            np.concatenate(roi_align_index_test)
        )
        return roi_align_index_tensor, roi_align_index_test_tensor

    def construct(self, img_data, img_metas, gt_bboxes, gt_labels, gt_valids):
        """
        construct the LibraRcnn Network.

        Args:
            img_data: input image data.
            img_metas: meta label of img.
            gt_bboxes (Tensor): get the value of bboxes.
            gt_labels (Tensor): get the value of labels.
            gt_valids (Tensor): get the valid part of bboxes.

        Returns:
            Tuple,tuple of output tensor (losses if training else predictions).
        """
        x = self.backbone(img_data)
        x = self.neck(x)
        if self.training:
            output = self.run_train(
                x, img_metas, gt_bboxes, gt_labels, gt_valids
            )
        else:
            output = self.run_test(x, img_metas)

        return output

    def run_train(self, feats, img_metas, gt_bboxes, gt_labels, gt_valids):
        """Run LibraRcnn loss calculation."""
        # Compute RPN loss.
        cls_score, bbox_pred = self.rpn_head(feats)
        (
            rpn_bbox_targets, rpn_bbox_weights, rpn_labels, rpn_label_weights
        ) = self.rpn_head.get_targets(
            gt_bboxes, self.gt_labels_stage1, gt_valids, self.anchor_list,
            img_metas
        )
        rpn_loss, rpn_cls_loss, rpn_reg_loss = self.rpn_head.loss(
            rpn_bbox_targets, rpn_bbox_weights, rpn_labels, rpn_label_weights,
            cls_score, bbox_pred
        )

        # Generate proposals
        proposal, proposal_mask = self.proposal_generator(
            cls_score, bbox_pred, self.anchor_list, img_metas
        )
        proposal = [ops.stop_gradient(p) for p in proposal]
        proposal_mask = [ops.stop_gradient(p) for p in proposal_mask]

        gt_labels = ops.cast(gt_labels, ms.int32)
        gt_valids = ops.cast(gt_valids, ms.int32)

        # Generate targets for rcnn and extract RoIs
        (
            bboxes_tuple, deltas_tuple, labels_tuple, mask_tuple
        ) = self.bbox_head.get_targets(
            gt_bboxes, gt_labels, gt_valids, proposal, proposal_mask
        )

        bbox_targets = ops.concat(deltas_tuple, axis=0)
        rcnn_labels = ops.concat(labels_tuple, axis=0)
        bbox_targets = ops.stop_gradient(bbox_targets)
        rcnn_labels = ops.stop_gradient(rcnn_labels)
        rcnn_labels = ops.cast(rcnn_labels, ms.int32)

        rcnn_masks = ops.concat(mask_tuple, axis=0)
        rcnn_masks = ops.stop_gradient(rcnn_masks)
        rcnn_mask_squeeze = ops.squeeze(ops.cast(rcnn_masks, ms.bool_))

        if self.train_batch_size > 1:
            bboxes_all = ops.concat(bboxes_tuple, axis=0)
        else:
            bboxes_all = bboxes_tuple[0]

        rois = ops.concat((self.roi_align_index_tensor, bboxes_all), axis=1)
        rois = ops.cast(rois, ms.float32)
        rois = ops.stop_gradient(rois)

        roi_feats = self.roi_align(
            rois,
            [
                ops.cast(feat, ms.float32)
                for feat in feats[:self.roi_align_test.num_levels]
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
            weights=rcnn_mask_squeeze
        )

        return (
            rpn_loss, rcnn_loss, rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss,
            rcnn_reg_loss
        )

    def run_test(self, feats, img_metas):
        """Run prediction calculation."""
        cls_score, bbox_pred = self.rpn_head(feats)
        proposal, proposal_mask = self.proposal_generator_test(
            cls_score, bbox_pred, self.anchor_list, img_metas
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

        res_bboxes, res_scores, res_mask = self.bbox_head.get_det_bboxes(
            cls_logits, reg_logits, rcnn_masks, bboxes_all, img_metas
        )

        return res_bboxes, res_scores, res_mask, None, None, None

    def set_train(self, mode=True):
        """Change training mode."""
        super().set_train(mode=mode)
        self.backbone.set_train(mode=mode)


class LibraRcnnInfer(nn.Cell):
    """LibraRCNN wrapper for inference."""

    def __init__(self, config):
        super().__init__()
        self.net = LibraRcnn(config)
        self.net.set_train(False)

    def construct(self, img_data, img_metas=None):
        """Make predictions."""
        output = self.net(img_data, img_metas, None, None, None)
        bboxes, labels, mask, _, _, _ = output
        return bboxes, labels, mask
