# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
"""RPN for LibraRCNN"""
from typing import Sequence

import numpy as np
import mindspore as ms

from mindspore import ops
from mindspore import nn
from mindspore import Tensor

from .sampling_builder import build_max_iou_random
from .. import Config


class RpnRegClsBlock(nn.Cell):
    """
    Rpn reg cls block for rpn layer

    Args:
        in_channels (int) - Input channels of shared convolution.
        feat_channels (int) - Output channels of shared convolution.
        num_anchors (int) - The anchor number.
        cls_out_channels (int) - Output channels of classification convolution.

    Returns:
        Tensor, output tensor.
    """
    def __init__(
            self,
            in_channels: int,
            feat_channels: int,
            num_anchors: int,
            cls_out_channels: int
    ):
        super(RpnRegClsBlock, self).__init__()
        self.rpn_conv = nn.Conv2d(
            in_channels, feat_channels, kernel_size=3, stride=1,
            pad_mode='same', has_bias=True
        )
        self.relu = nn.ReLU()

        self.rpn_cls = nn.Conv2d(
            feat_channels, num_anchors * cls_out_channels, kernel_size=1,
            pad_mode='valid', has_bias=True
        )
        self.rpn_reg = nn.Conv2d(
            feat_channels, num_anchors * 4, kernel_size=1, pad_mode='valid',
            has_bias=True
        )

    def construct(self, x):
        x = self.relu(self.rpn_conv(x))

        x1 = self.rpn_cls(x)
        x2 = self.rpn_reg(x)

        return x1, x2


class RPN(nn.Cell):
    """
    ROI proposal network..

    Args:
        num_bboxes (int): Common number of predicted bboxes,
        feature_shapes (Sequence): list of feature maps sizes,
        batch_size (int): train batch size,
        in_channels (int): number of input channels,
        feat_channels (int): number of intermediate channels,
        cls_out_channels (int): number of classes,
        num_anchors (int): number of anchors,
        cls_loss (Config): information about classification criterion,
        reg_loss (Config): information about regression criterion,
        bbox_assign_sampler (Config): information about assigner and sampler,
        num_gts (int): number of gts,
        target_means (Sequence): bbox coder parameter,
        target_stds (Sequence): bbox coder parameter,

    Returns:
        Tuple, tuple of output tensor.

    Examples:
        RPN(config=config, batch_size=2, in_channels=256, feat_channels=1024,
            num_anchors=3, cls_out_channels=512)
    """
    def __init__(
            self,
            num_bboxes: int,
            feature_shapes: Sequence,
            batch_size: int,
            in_channels: int,
            feat_channels: int,
            cls_out_channels: int,
            num_anchors: int,
            cls_loss: Config,
            reg_loss: Config,
            bbox_assign_sampler: Config,
            num_gts: int,
            target_means: Sequence = (0., 0., 0., 0.),
            target_stds: Sequence = (1., 1., 1., 1.),
    ):
        super(RPN, self).__init__()
        self.dtype = np.float32
        self.ms_type = ms.float32
        self.num_bboxes = num_bboxes
        self.feature_anchor_shape = ()
        self.slice_index = (0,)
        index = 0
        for shape in feature_shapes:
            self.slice_index += (
                self.slice_index[index] + shape[0] * shape[1] * num_anchors,
            )
            self.feature_anchor_shape += (
                shape[0] * shape[1] * num_anchors * batch_size,
            )
            index += 1

        self.num_anchors = num_anchors
        self.batch_size = batch_size
        self.num_layers = 5
        self.real_ratio = ops.ones((1, 1), self.ms_type)

        self.rpn_convs_layer = self._make_rpn_layer(
            in_channels, feat_channels, num_anchors,
            cls_out_channels
        )

        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.concat = ops.Concat(axis=0)
        self.fill = ops.Fill()
        self.placeh1 = ops.ones((1,), self.ms_type)

        self.trans_shape = (0, 2, 3, 1)

        self.reshape_shape_reg = (-1, 4)
        self.reshape_shape_cls = (-1,)
        self.rpn_loss_reg_weight = Tensor(
            np.array(reg_loss.weight).astype(self.dtype)
        )
        self.rpn_loss_cls_weight = Tensor(
            np.array(cls_loss.weight).astype(self.dtype)
        )
        self.num_expected_total = Tensor(
            np.array(
                bbox_assign_sampler.num_expected_neg * self.batch_size
            ).astype(self.dtype)
        )
        self.targets_generator = build_max_iou_random(
            num_bboxes=self.num_bboxes,
            num_gts=num_gts,
            pos_num_expected=bbox_assign_sampler.num_expected_pos,
            neg_num_expected=bbox_assign_sampler.num_expected_neg,
            neg_iou_thr=bbox_assign_sampler.neg_iou_thr,
            pos_iou_thr=bbox_assign_sampler.pos_iou_thr,
            min_pos_iou=bbox_assign_sampler.min_pos_iou,
            neg_pos_ub=bbox_assign_sampler.neg_pos_ub,
            add_gt_as_proposals=False,
            match_low_quality=bbox_assign_sampler.match_low_quality,
            target_means=target_means,
            target_stds=target_stds,
            rcnn_mode=False
        )
        self.CheckValid = ops.CheckValid()
        self.sum_loss = ops.ReduceSum()
        self.loss_cls = ops.SigmoidCrossEntropyWithLogits()
        self.loss_bbox = nn.L1Loss(reduction='none')
        self.squeeze = ops.Squeeze()
        self.cast = ops.Cast()
        self.tile = ops.Tile()
        self.zeros_like = ops.ZerosLike()
        self.total_loss = ops.zeros((1,), self.ms_type)
        self.clsloss = ops.zeros((1,), self.ms_type)
        self.regloss = ops.zeros((1,), self.ms_type)

    def _make_rpn_layer(
            self, in_channels, feat_channels, num_anchors,
            cls_out_channels
    ):
        """
        make rpn layer for rpn proposal network

        Args:
        in_channels (int) - Input channels of shared convolution.
        feat_channels (int) - Output channels of shared convolution.
        num_anchors (int) - The anchor number.
        cls_out_channels (int) - Output channels of classification convolution.

        Returns:
        List, list of RpnRegClsBlock cells.
        """
        rpn_reg_cls_block = RpnRegClsBlock(
            in_channels, feat_channels, num_anchors, cls_out_channels,
        )

        return rpn_reg_cls_block

    def construct(
            self, inputs
    ):
        rpn_cls_score_total = []
        rpn_bbox_pred_total = []

        for i in range(self.num_layers):
            x1, x2 = self.rpn_convs_layer(inputs[i])

            rpn_cls_score_total.append(x1)
            rpn_bbox_pred_total.append(x2)

        return rpn_cls_score_total, rpn_bbox_pred_total

    def get_targets(
            self, gt_bboxes, gt_labels, gt_valids, anchor_list, img_metas
    ):
        """Get targets for rpn."""
        bbox_targets, bbox_weights, labels, label_weights = [], [], [], []
        for i in range(self.batch_size):
            multi_level_flags, anchor_list_tuple = [], []

            for j in range(self.num_layers):
                res = self.cast(
                    self.CheckValid(
                        anchor_list[j], self.squeeze(img_metas[i:i + 1:1, ::])
                    ),
                    ms.int32
                )
                multi_level_flags.append(res)
                anchor_list_tuple.append(anchor_list[j])

            valid_flag_list = self.concat(multi_level_flags)
            anchor_using_list = self.concat(anchor_list_tuple)

            gt_bboxes_i = self.squeeze(gt_bboxes[i])
            gt_labels_i = self.squeeze(gt_labels[i])
            gt_valids_i = self.squeeze(gt_valids[i])

            (
                bbox_target, bbox_weight, label, label_weight
            ) = self.targets_generator(
                gt_bboxes_i, gt_labels_i, self.cast(valid_flag_list, ms.bool_),
                anchor_using_list, gt_valids_i
            )

            bbox_target = self.cast(bbox_target, self.ms_type)
            bbox_weight = self.cast(bbox_weight, self.ms_type)
            label = self.cast(label, self.ms_type)
            label_weight = self.cast(label_weight, self.ms_type)

            for j in range(self.num_layers):
                begin = self.slice_index[j]
                end = self.slice_index[j + 1]
                stride = 1
                bbox_targets.append(bbox_target[begin:end:stride, ::])
                bbox_weights.append(bbox_weight[begin:end:stride])
                labels.append(label[begin:end:stride])
                label_weights.append(label_weight[begin:end:stride])

        return bbox_targets, bbox_weights, labels, label_weights

    def loss(
            self, bbox_targets, bbox_weights, labels, label_weights,
            rpn_cls_score_total, rpn_bbox_pred_total
    ):
        """Compute RPN loss."""
        rpn_cls_score = []
        rpn_bbox_pred = []
        for i in range(self.num_layers):
            x1 = rpn_cls_score_total[i]
            x1 = self.transpose(x1, self.trans_shape)
            x1 = self.reshape(x1, self.reshape_shape_cls)
            x2 = rpn_bbox_pred_total[i]
            x2 = self.transpose(x2, self.trans_shape)
            x2 = self.reshape(x2, self.reshape_shape_reg)
            rpn_cls_score.append(x1)
            rpn_bbox_pred.append(x2)

        loss, clsloss, regloss = self.total_loss, self.clsloss, self.regloss
        res = []
        for i in range(self.num_layers):
            bbox_target_using, bbox_weight_using = [], []
            label_using, label_weight_using = [], []

            for j in range(self.batch_size):
                bbox_target_using.append(bbox_targets[i + (self.num_layers * j)])
                bbox_weight_using.append(bbox_weights[i + (self.num_layers * j)])
                label_using.append(labels[i + (self.num_layers * j)])
                label_weight_using.append(
                    label_weights[i + (self.num_layers * j)]
                )

            bbox_target_with_batchsize = self.concat(bbox_target_using)
            bbox_weight_with_batchsize = self.concat(bbox_weight_using)
            label_with_batchsize = self.concat(label_using)
            label_weight_with_batchsize = self.concat(label_weight_using)

            # stop
            bbox_target_ = ops.stop_gradient(bbox_target_with_batchsize)
            bbox_weight_ = ops.stop_gradient(bbox_weight_with_batchsize)
            label_ = ops.stop_gradient(label_with_batchsize)
            label_weight_ = ops.stop_gradient(label_weight_with_batchsize)

            cls_score_i = self.cast(rpn_cls_score[i], self.ms_type)
            reg_score_i = self.cast(rpn_bbox_pred[i], self.ms_type)

            loss_cls = self.loss_cls(cls_score_i, label_)
            loss_cls_item = loss_cls * label_weight_
            loss_cls_item = self.sum_loss(loss_cls_item,
                                          (0,)) / self.num_expected_total

            loss_reg = self.loss_bbox(reg_score_i, bbox_target_)
            bbox_weight_ = self.tile(
                self.reshape(bbox_weight_, (self.feature_anchor_shape[i], 1)),
                (1, 4))

            loss_reg = loss_reg * bbox_weight_

            loss_reg_item = self.sum_loss(loss_reg, (1,))
            loss_reg_item = self.sum_loss(loss_reg_item,
                                          (0,)) / self.num_expected_total

            loss_total = (self.rpn_loss_cls_weight * loss_cls_item +
                          self.rpn_loss_reg_weight * loss_reg_item)

            loss += loss_total
            clsloss += loss_cls_item
            regloss += loss_reg_item
            res = loss, clsloss, regloss

        return res
