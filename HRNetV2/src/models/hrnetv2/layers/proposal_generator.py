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
"""Proposal generator generates proposed bboxes based on RPNHead predictions.
"""
from typing import Sequence, Tuple

import numpy as np
import mindspore as ms

from mindspore import nn
from mindspore import ops
from mindspore import Tensor

from .bbox_coder import DeltaXYWHBBoxCoder


class Proposal(nn.Cell):
    """
    Proposal subnet. Used to generate the most confident bbox based on RPN
    block predictions.
    """

    def __init__(
            self,
            batch_size: int,
            num_classes: int,
            feature_shapes: Sequence[Tuple[int, int]],
            bbox_coder: DeltaXYWHBBoxCoder,
            use_sigmoid_cls: bool = True,
            nms_pre: int = 2000,
            iou_threshold: float = 0.5,
            min_bbox_size: int = 0,
            max_per_img: int = 1000,
            num_levels: int = 5,
    ):
        """
        Init proposal generator.

        Args:
            batch_size (int): Batch size.
            num_classes (int): Number of classes.
            use_sigmoid_cls (bool): Whether use sigmoid.
            feature_shapes (Sequence[Tuple[int, int]]): List of
                feature maps sizes.
            nms_pre (int): Max number of samples.
            iou_threshold (float): IoU threshold for NMS algorithm.
            max_per_img (int): Max number of output samples.
            num_levels (int): Number feature maps.

        Returns:
            Tuple, tuple of output tensor,(proposal, mask).
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.use_sigmoid_cls = use_sigmoid_cls
        self.min_bbox_size = min_bbox_size

        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
            self.activation = ops.Sigmoid()
            self.reshape_shape = (-1, 1)
        else:
            self.cls_out_channels = num_classes
            self.activation = ops.Softmax(axis=1)
            self.reshape_shape = (-1, 2)

        if self.cls_out_channels <= 0:
            raise ValueError(f'num_classes={num_classes} is too small')

        self.num_pre = nms_pre
        self.nms_thr = iou_threshold
        self.max_num = max_per_img
        self.num_levels = num_levels

        # Op Define
        self.cast = ops.Cast()

        self.feature_shapes = feature_shapes

        self.transpose_shape = (1, 2, 0)

        self.nms = ops.NMSWithMask(self.nms_thr)
        self.split = ops.Split(axis=1, output_num=5)

        self.dtype = np.float32
        self.ms_type = ms.float32

        self.topK_stage1 = ()
        self.topK_shape = ()
        total_max_topk_input = 0

        for shp in self.feature_shapes:
            k_num = min(self.num_pre, (shp[0] * shp[1] * 3))
            total_max_topk_input += k_num
            self.topK_stage1 += (k_num,)
            self.topK_shape += ((k_num, 1),)

        self.topK_shape_stage2 = (self.max_num, 1)
        self.min_float_num = -65500.0
        self.topK_mask = Tensor(
            self.min_float_num * np.ones(total_max_topk_input, np.float32)
        )
        self.bbox_coder = bbox_coder

    def construct(
            self,
            rpn_cls_score_total: Sequence[ms.Tensor],
            rpn_bbox_pred_total: Sequence[ms.Tensor],
            anchor_list: Sequence[ms.Tensor],
            img_metas: ms.Tensor
    ) -> Tuple[Sequence[ms.Tensor], Sequence[ms.Tensor]]:
        """Use RPNHead output and generate proposals.

        Args:
            rpn_cls_score_total (Sequence[ms.Tensor]): List of RPN
                classification predictions. Number of tensors equal to number
                of feature maps levels. Each tensor shape
                (n, num_anchors, self.feature_shapes[i][0],
                self.feature_shapes[i][1]).
            rpn_bbox_pred_total (Sequence[ms.Tensor]): List of RPN
                localization predictions. Number of tensors equal to number
                of feature maps levels. Each tensor shape
                (n, anchors_num * 4, self.feature_shapes[i][0],
                self.feature_shapes[i][1]).
            anchor_list (Sequence[ms.Tensor]): List of anchors, corresponded
                to RPN predictions. Number of tensors equal to number
                of feature maps levels.
            img_metas: (ms.Tensor): List of images meta information.

        Returns:
            Sequence[ms.Tensor]: Generated proposed bounding boxes lists for
                each image. Shape of each tensor (self.max_num, 5)
        """
        proposals_tuple = ()
        masks_tuple = ()
        for img_id in range(self.batch_size):
            cls_score_list = ()
            bbox_pred_list = ()
            for i in range(self.num_levels):
                rpn_cls_score_i = ops.squeeze(
                    rpn_cls_score_total[i][img_id:img_id+1:1, ::, ::, ::]
                )
                rpn_bbox_pred_i = ops.squeeze(
                    rpn_bbox_pred_total[i][img_id:img_id+1:1, ::, ::, ::]
                )

                cls_score_list = cls_score_list + (rpn_cls_score_i,)
                bbox_pred_list = bbox_pred_list + (rpn_bbox_pred_i,)
            shape = img_metas[img_id][4:6]
            proposals, masks = self.get_bboxes_single(
                cls_score_list, bbox_pred_list, anchor_list, shape
            )
            proposals_tuple += (proposals,)
            masks_tuple += (masks,)
        return proposals_tuple, masks_tuple

    def get_bboxes_single(
            self,
            cls_scores: Sequence[ms.Tensor],
            bbox_preds: Sequence[ms.Tensor],
            mlvl_anchors: Sequence[ms.Tensor],
            shape: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """Get proposal bounding boxes for single image in batch.

        Args:
            cls_scores (Sequence[ms.Tensor]): List of RPN classification
                predictions for one image in batch.
            bbox_preds (Sequence[ms.Tensor]): List of RPN localization
                predictions for one image in batch.
            mlvl_anchors (Sequence[ms.Tensor]): List of anchors, corresponded
                to RPN predictions. Number of tensors equal to number
                of feature maps levels.
            shape (ms.Tensor): Cropping values for image (h and w).

        Returns:
            ms.Tensor: Generated proposals for image.
            ms.Tensor: Mask that show what proposal bounding boxes are valid.
        """
        mlvl_proposals = ()
        mlvl_mask = ()
        for idx in range(self.num_levels):
            rpn_cls_score = ops.transpose(
                cls_scores[idx], self.transpose_shape
            )
            rpn_bbox_pred = ops.transpose(
                bbox_preds[idx], self.transpose_shape
            )
            anchors = mlvl_anchors[idx]

            rpn_cls_score = ops.reshape(rpn_cls_score, self.reshape_shape)
            rpn_cls_score = self.activation(rpn_cls_score)
            rpn_cls_score_process = self.cast(
                ops.squeeze(rpn_cls_score[::, 0::]), self.ms_type
            )

            rpn_bbox_pred_process = self.cast(
                ops.reshape(rpn_bbox_pred, (-1, 4)), self.ms_type
            )

            scores = rpn_cls_score_process
            bboxes = rpn_bbox_pred_process

            if self.num_pre > 0 and rpn_cls_score.shape[0] > self.num_pre:
                scores_sorted, topk_inds = ops.topk(
                    rpn_cls_score_process, self.topK_stage1[idx]
                )

                topk_inds = ops.reshape(topk_inds, self.topK_shape[idx])

                bboxes_sorted = ops.gather_nd(rpn_bbox_pred_process, topk_inds)
                anchors_sorted = self.cast(
                    ops.gather_nd(anchors, topk_inds), self.ms_type
                )
                scores_sorted = ops.reshape(
                    scores_sorted, self.topK_shape[idx]
                )
                anchors, scores, bboxes = (
                    anchors_sorted, scores_sorted, bboxes_sorted
                )
            else:
                scores = ops.reshape(scores, (-1, 1))
            proposals_decode = self.bbox_coder.decode(
                anchors, bboxes, shape
            )

            proposals_decode = ops.cat((proposals_decode, scores), axis=1)
            proposals, _, mask_valid = self.nms(proposals_decode)
            mlvl_proposals = mlvl_proposals + (proposals,)
            mlvl_mask = mlvl_mask + (mask_valid,)

        proposals = ops.cat(mlvl_proposals, axis=0)
        masks = ops.cat(mlvl_mask, axis=0)

        x1, y1, x2, y2, scores = self.split(proposals)
        w = x2 - x1
        h = y2 - y1
        min_bbox_mask = ops.logical_and(
            ops.ge(w, self.min_bbox_size),
            ops.ge(h, self.min_bbox_size)
        ).reshape(-1)
        masks = ops.logical_and(masks, min_bbox_mask)
        scores = ops.squeeze(scores)
        topk_mask = self.cast(self.topK_mask, self.ms_type)

        scores_using = ops.select(masks, scores, topk_mask)

        _, topk_inds = ops.topk(scores_using, self.max_num)

        topk_inds = ops.reshape(topk_inds, self.topK_shape_stage2)
        proposals = ops.gather_nd(proposals, topk_inds)
        masks = ops.gather_nd(masks, topk_inds)
        return proposals, masks
