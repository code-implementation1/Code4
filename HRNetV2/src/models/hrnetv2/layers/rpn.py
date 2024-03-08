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
"""RPNHead for RCNN models."""
from functools import reduce
from typing import Sequence, Tuple
import numpy as np
import mindspore as ms

from mindspore import nn
from mindspore import ops
from mindspore import Tensor

from src.config import Config
from .conv_module import ConvModule
from .proposal_generator import Proposal


class RpnRegClsBlock(nn.Cell):
    """
    RPN Regression/Classification block for rpn layer

    Args:
        in_channels (int): Input channels of shared convolution.
        feat_channels (int): Output channels of shared convolution.
        num_anchors (int): The anchor number.
        cls_out_channels (int): Output channels of classification convolution.
        num_convs (int): Number of shared convolutions.

    Returns:
        Tensor, output tensor.
    """

    def __init__(
            self,
            in_channels: int,
            feat_channels: int,
            num_anchors: int,
            cls_out_channels: int,
            num_convs: int = 1
    ):
        super(RpnRegClsBlock, self).__init__()
        if num_convs > 1:
            rpn_convs = []
            for i in range(num_convs):
                if i == 0:
                    in_channels = in_channels
                else:
                    in_channels = feat_channels
                rpn_convs.append(
                    ConvModule(
                        in_channels,
                        feat_channels,
                        3,
                        padding_mode='same',
                        inplace=False
                    )
                )
            self.rpn_conv = nn.SequentialCell(*rpn_convs)
        else:
            self.rpn_conv = nn.Conv2d(
                in_channels, feat_channels, 3, has_bias=True, stride=1,
                pad_mode='same'
            )
        self.rpn_cls = nn.Conv2d(
            feat_channels, num_anchors * cls_out_channels,
            1, has_bias=True, pad_mode='valid'
        )
        self.rpn_reg = nn.Conv2d(
            feat_channels, num_anchors * 4, 1,
            has_bias=True, pad_mode='valid'
        )
        self.relu = ops.ReLU()

    def construct(self, x: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
        """Forward RPN layer."""
        x = self.relu(self.rpn_conv(x))

        x1 = self.rpn_cls(x)
        x2 = self.rpn_reg(x)

        return x1, x2


class RPNHead(nn.Cell):
    """ROI proposal network.

    Args:
        feature_shapes (Sequence): List of feature maps sizes.
        train_batch_size (int): Number of images per training step.
        test_batch_size (int): Number of images per test step.
        num_gts (int): Max number of GT bboxes.
        anchor_generator (nn.Cell): Block that generates anchors.
        bbox_coder (nn.Cell): Bounding box coder.
        loss_cls (nn.Cell): Classification loss block.
        loss_bbox (nn.Cell): Regression loss block.
        targets_generator (nn.Cell): Block that generates targets. Contain
            assigner, sampler and bbox_coder (encode bounding boxes).
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes.
        num_convs (int): Number of shared convolution layers.
        feat_channels (int): Number of channels in hidden feature maps.
        loss_cls_weight (float): Classification loss weight.
        loss_bbox_weight (float): Regression loss weight.
        train_cfg (Config): Model training configurations.
        test_cfg (Config): Model inference configurations.

    Examples:
        >>> anchor_generator = AnchorGenerator(
        ...     scales=[8],
        ...     ratios=[0.5, 1.0, 2.0],
        ...     strides=[4, 8, 16, 32, 64]
        ... )
        >>> bbox_coder = DeltaXYWHBBoxCoder()
        >>> feature_shapes = [(2 ** (i + 4), ) * 2 for i in range(5)]
        >>> loss_cls = nn.SoftmaxCrossEntropyWithLogits(reduction='none')
        >>> loss_bbox = nn.L1Loss(reduction='none')
        >>> rpn = RPNHead(
        ...     feature_shapes=feature_shapes,
        ...     anchor_generator=anchor_generator,
        ...     bbox_coder=bbox_coder,
        ...     loss_cls=loss_cls,
        ...     loss_bbox=loss_bbox,
        ...     targets_generator=None,
        ...     in_channels=256
        ... )
        >>> features = [
        ...     ms.Tensor(np.random.rand(1, 256, *fs), ms.float32)
        ...     for fs in feature_shapes
        ... ]
        >>> cls_res, bbox_res = rpn(features)
        >>> len(cls_res), len(bbox_res)
        (5, 5)
        >>> [a.shape for a in cls_res]
        [(1, 3, 16, 16), (1, 3, 32, 32), (1, 3, 64, 64), (1, 3, 128, 128), (1, 3, 256, 256)]
        >>> [a.shape for a in bbox_res]
        [(1, 12, 16, 16), (1, 12, 32, 32), (1, 12, 64, 64), (1, 12, 128, 128), (1, 12, 256, 256)]
    """

    def __init__(
            self,
            feature_shapes: Sequence[Tuple[int, int]],
            anchor_generator: nn.Cell,
            bbox_coder: nn.Cell,
            loss_cls: nn.Cell,
            loss_bbox: nn.Cell,
            targets_generator: nn.Cell,
            in_channels: int,
            train_batch_size: int = 4,
            test_batch_size: int = 1,
            num_gts: int = 100,
            num_classes: int = 1,
            num_convs: int = 1,
            feat_channels: int = 256,
            loss_cls_weight: float = 1.0,
            loss_bbox_weight: float = 1.0,
            train_cfg: Config = Config(
                dict(
                    rpn_proposal=dict(
                        nms_pre=2000, max_per_img=1000,
                        iou_threshold=0.7,
                        min_bbox_size=0
                    ),
                    rpn=dict(
                        assigner=dict(
                            pos_iou_thr=0.7,
                            neg_iou_thr=0.3,
                            min_pos_iou=0.3,
                            match_low_quality=True
                        ),
                        sampler=dict(
                            num=256,
                            pos_fraction=0.5,
                            neg_pos_ub=5,
                            add_gt_as_proposals=False
                        )
                    )
                )
            ),
            test_cfg: Config = Config(
                dict(
                    rpn=dict(
                        nms_pre=1000, max_per_img=1000,
                        iou_threshold=0.7, min_bbox_size=0
                    )
                )
            ),
    ):
        """Init RPNHead."""
        super(RPNHead, self).__init__()
        self.dtype = np.float32
        self.ms_type = ms.float32
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_convs = num_convs
        self.feat_channels = feat_channels
        self.num_gts = num_gts
        self.train_batch_size = train_batch_size
        self.val_batch_size = test_batch_size

        # generate anchors
        self.anchor_generator = anchor_generator
        self.feature_shapes = feature_shapes
        self.anchor_list = self.anchor_generator.grid_priors(
            self.feature_shapes
        )
        self.num_anchors = self.anchor_generator.num_base_priors[0]
        self.num_bboxes = sum(
            [
                reduce(lambda x, y: x * y, anchors.shape, 1)
                for anchors in self.anchor_list
            ]
        ) // 4

        self.feature_anchor_shape = ()

        self.slice_index = (0,)
        index = 0
        for shape in feature_shapes:
            self.slice_index += (
                self.slice_index[index] +
                shape[0] * shape[1] * self.num_anchors,
            )
            self.feature_anchor_shape += (
                shape[0] * shape[1] * self.num_anchors * train_batch_size,
            )
            index += 1

        self.num_layers = 5
        self.real_ratio = ops.ones((1, 1), self.ms_type)

        self.rpn_convs_layer = self._make_rpn_layer()

        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.concat = ops.Concat(axis=0)
        self.fill = ops.Fill()
        self.placeh1 = ops.ones((1,), self.ms_type)

        self.trans_shape = (0, 2, 3, 1)

        self.reshape_shape_reg = (-1, 4)
        self.reshape_shape_cls = (-1,)

        self.loss_cls = loss_cls
        self.loss_bbox = loss_bbox
        self.rpn_loss_reg_weight = Tensor(
            np.array(loss_bbox_weight).astype(self.dtype)
        )
        self.rpn_loss_cls_weight = Tensor(
            np.array(loss_cls_weight).astype(self.dtype)
        )
        self.bbox_coder = bbox_coder
        self.targets_generator = targets_generator

        self.num_expected_total = Tensor(
            np.array(
                train_cfg.rpn.sampler.num * self.train_batch_size
            ).astype(self.dtype)
        )

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        (
            self.proposal_generator, self.proposal_generator_test
        ) = self.create_proposal_generator()

        self.CheckValid = ops.CheckValid()
        self.sum_loss = ops.ReduceSum()

        self.squeeze = ops.Squeeze()
        self.cast = ops.Cast()
        self.tile = ops.Tile()
        self.zeros_like = ops.ZerosLike()
        self.total_loss = ops.zeros((1,), self.ms_type)
        self.clsloss = ops.zeros((1,), self.ms_type)
        self.regloss = ops.zeros((1,), self.ms_type)

    def create_proposal_generator(self):
        """Create proposal generator."""
        proposal_generator = Proposal(
            batch_size=self.train_batch_size,
            num_classes=self.num_classes + 1,
            feature_shapes=self.feature_shapes,
            num_levels=len(self.feature_shapes),
            bbox_coder=self.bbox_coder,
            **self.train_cfg.rpn_proposal.as_dict()
        )
        proposal_generator_test = Proposal(
            batch_size=self.val_batch_size,
            num_classes=self.num_classes + 1,
            feature_shapes=self.feature_shapes,
            num_levels=len(self.feature_shapes),
            bbox_coder=self.bbox_coder,
            **self.test_cfg.rpn.as_dict()
        )
        return proposal_generator, proposal_generator_test

    def _make_rpn_layer(self):
        """Make rpn layer for rpn proposal network."""
        rpn_reg_cls_block = RpnRegClsBlock(
            in_channels=self.in_channels,
            feat_channels=self.feat_channels,
            num_anchors=self.num_anchors,
            cls_out_channels=self.num_classes,
            num_convs=self.num_convs
        )

        return rpn_reg_cls_block

    def construct(
            self, inputs: Sequence[ms.Tensor]
    ) -> Tuple[Sequence[ms.Tensor], Sequence[ms.Tensor]]:
        """RPNHead forward method.

        Args:
            inputs (Sequence[ms.Tensor]): List of input feature maps for each
            layer. Shape of each tensor
            (n, self.in_channels, self.feature_shapes[i][0],
            self.feature_shapes[i][1]).

        Returns:
            Sequence[ms.Tensor]:
                Classification predictions. Each tensor shape
                (n, self.num_anchors, self.feature_shapes[i][0],
                self.feature_shapes[i][1]).
            Sequence[ms.Tensor]:
                Localization predictions. Each tensor shape
                (n, self.num_anchors * 4, self.feature_shapes[i][0],
                self.feature_shapes[i][1]).
        """
        rpn_cls_score_total = []
        rpn_bbox_pred_total = []

        for i in range(self.num_layers):
            x1, x2 = self.rpn_convs_layer(inputs[i])

            rpn_cls_score_total.append(x1)
            rpn_bbox_pred_total.append(x2)

        return rpn_cls_score_total, rpn_bbox_pred_total

    def get_targets(
            self, gt_bboxes, gt_labels, gt_valids, img_metas
    ):
        """Get targets for rpn."""
        bbox_targets, bbox_weights, labels, label_weights = [], [], [], []
        for i in range(self.train_batch_size):
            multi_level_flags, anchor_list_tuple = [], []
            img_shape = ops.concat((img_metas[i, 4:6], ops.ones((1,))))
            for j in range(self.num_layers):
                res = self.cast(
                    self.CheckValid(self.anchor_list[j], img_shape), ms.int32
                )
                multi_level_flags.append(res)
                anchor_list_tuple.append(self.anchor_list[j])

            valid_flag_list = self.concat(multi_level_flags)
            anchor_using_list = self.concat(anchor_list_tuple)

            gt_bboxes_i = self.squeeze(gt_bboxes[i])
            gt_labels_i = self.squeeze(gt_labels[i])
            gt_valids_i = self.squeeze(gt_valids[i])

            (
                bbox_target, bbox_weight, label, label_weight
            ) = self.targets_generator(
                gt_bboxes_i=gt_bboxes_i,
                gt_labels_i=gt_labels_i,
                bboxes=anchor_using_list,
                valid_mask=self.cast(valid_flag_list, ms.bool_),
                gt_valids=gt_valids_i
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

            for j in range(self.train_batch_size):
                bbox_target_using.append(
                    bbox_targets[i + (self.num_layers * j)])
                bbox_weight_using.append(
                    bbox_weights[i + (self.num_layers * j)])
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

            loss_cls_item = self.rpn_loss_cls_weight * loss_cls_item
            loss_reg_item = self.rpn_loss_reg_weight * loss_reg_item

            loss_total = loss_cls_item + loss_reg_item

            loss += loss_total
            clsloss += loss_cls_item
            regloss += loss_reg_item
            res = loss, clsloss, regloss

        return res

    def get_bboxes(self, cls_score, bbox_pred, img_metas):
        """Use RPNHead output and generate proposals.

        Args:
            cls_score (Sequence[ms.Tensor]): List of RPN
                classification predictions. Number of tensors equal to number
                of feature maps levels. Each tensor shape
                (n, num_anchors, self.feature_shapes[i][0],
                self.feature_shapes[i][1]).
            bbox_pred (Sequence[ms.Tensor]): List of RPN
                localization predictions. Number of tensors equal to number
                of feature maps levels. Each tensor shape
                (n, anchors_num * 4, self.feature_shapes[i][0],
                self.feature_shapes[i][1]).
            img_metas: (ms.Tensor): List of images meta information.

        Returns:
            Sequence[ms.Tensor]: Generated proposed bounding boxes lists for
                each image. Shape of each tensor (self.max_num, 5)
        """
        if self.training:
            outputs = self.proposal_generator(
                cls_score, bbox_pred, self.anchor_list, img_metas
            )
        else:
            outputs = self.proposal_generator_test(
                cls_score, bbox_pred, self.anchor_list, img_metas
            )
        return outputs
