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
"""LibraRcnn proposal generator."""
from typing import Sequence, Tuple

import numpy as np
import mindspore as ms

from mindspore import ops
from mindspore import nn
from mindspore import Tensor


from .bbox_coder import DeltaXYWHBBoxCoder


class Proposal(nn.Cell):
    """
    Proposal subnet. Used to generate the most confident bbox based on RPN
    block predictions.

    Args:
        batch_size (int): batch size,
        num_classes (int): number of classes,
        use_sigmoid_cls (bool): whether use sigmoid,
        feature_shapes (Sequence[Tuple[int, int]]): list of feature maps sizes,
        img_height (int): input image height,
        img_width (int): input image width,
        nms_pre (int): max number of samples,
        nms_thr (float): num threshold,
        max_num (int): max number of output samples,
        num_levels (int): number feature maps,
        target_means (Sequence[int]): bbox coder parameter,
        target_stds (Sequence[int]): bbox coder parameter

    Returns:
        Tuple, tuple of output tensor,(proposal, mask).
    """
    def __init__(
            self,
            batch_size: int,
            num_classes: int,
            use_sigmoid_cls: bool,
            feature_shapes: Sequence[Tuple[int, int]],
            img_height: int,
            img_width: int,
            nms_pre: int,
            nms_thr: float,
            max_num: int,
            num_levels: int = 5,
            target_means: Sequence[int] = (.0, .0, .0, .0),
            target_stds: Sequence[int] = (1.0, 1.0, 1.0, 1.0)
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.means_tensor = ms.Tensor(target_means).reshape(-1, 4)
        self.stds_tensor = ms.Tensor(target_stds).reshape(-1, 4)
        self.wh_ratio_clip = ms.Tensor(16 / 1000)
        self.img_shape = (img_height, img_width)
        self.use_sigmoid_cls = use_sigmoid_cls
        self.min_bbox_size = 0

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
        self.nms_thr = nms_thr
        self.max_num = max_num
        self.num_levels = num_levels

        # Op Define
        self.squeeze = ops.Squeeze()
        self.reshape = ops.Reshape()
        self.cast = ops.Cast()

        self.feature_shapes = feature_shapes

        self.transpose_shape = (1, 2, 0)

        self.nms = ops.NMSWithMask(self.nms_thr)
        self.concat_axis0 = ops.Concat(axis=0)
        self.concat_axis1 = ops.Concat(axis=1)
        self.split = ops.Split(axis=1, output_num=5)
        self.min = ops.Minimum()
        self.gatherND = ops.GatherNd()
        self.slice = ops.Slice()
        self.select = ops.Select()
        self.greater = ops.Greater()
        self.transpose = ops.Transpose()
        self.tile = ops.Tile()

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

        self.topKv2 = ops.TopK(sorted=True)
        self.topK_shape_stage2 = (self.max_num, 1)
        self.min_float_num = -65500.0
        self.topK_mask = Tensor(
            self.min_float_num * np.ones(total_max_topk_input, np.float32)
        )
        self.bbox_coder = DeltaXYWHBBoxCoder(
            target_means=target_means, target_stds=target_stds
        )

    def construct(
            self,
            rpn_cls_score_total: Sequence[ms.Tensor],
            rpn_bbox_pred_total: Sequence[ms.Tensor],
            anchor_list: Sequence[ms.Tensor],
            img_metas: ms.Tensor
    ) -> Tuple[Sequence[ms.Tensor], Sequence[ms.Tensor]]:
        proposals_tuple = ()
        masks_tuple = ()
        for img_id in range(self.batch_size):
            cls_score_list = ()
            bbox_pred_list = ()
            for i in range(self.num_levels):
                rpn_cls_score_i = self.squeeze(
                    rpn_cls_score_total[i][img_id:img_id+1:1, ::, ::, ::]
                )
                rpn_bbox_pred_i = self.squeeze(
                    rpn_bbox_pred_total[i][img_id:img_id+1:1, ::, ::, ::]
                )

                cls_score_list = cls_score_list + (rpn_cls_score_i,)
                bbox_pred_list = bbox_pred_list + (rpn_bbox_pred_i,)
            if self.training:
                shape = img_metas[img_id][:2]
            else:
                shape = img_metas[img_id][2:4]
            proposals, masks = self.get_bboxes_single(
                cls_score_list, bbox_pred_list, anchor_list, shape
            )
            proposals_tuple += (proposals,)
            masks_tuple += (masks,)
        return proposals_tuple, masks_tuple

    def get_bboxes_single(
            self,
            cls_scores: ms.Tensor,
            bbox_preds: ms.Tensor,
            mlvl_anchors: Sequence[ms.Tensor],
            shape: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """Get proposal bounding boxes."""
        mlvl_proposals = ()
        mlvl_mask = ()
        for idx in range(self.num_levels):
            rpn_cls_score = self.transpose(
                cls_scores[idx], self.transpose_shape
            )
            rpn_bbox_pred = self.transpose(
                bbox_preds[idx], self.transpose_shape
            )
            anchors = mlvl_anchors[idx]

            rpn_cls_score = self.reshape(rpn_cls_score, self.reshape_shape)
            rpn_cls_score = self.activation(rpn_cls_score)
            rpn_cls_score_process = self.cast(
                self.squeeze(rpn_cls_score[::, 0::]), self.ms_type
            )

            rpn_bbox_pred_process = self.cast(
                self.reshape(rpn_bbox_pred, (-1, 4)), self.ms_type
            )

            scores = rpn_cls_score_process
            bboxes = rpn_bbox_pred_process

            if self.num_pre > 0 and rpn_cls_score.shape[0] > self.num_pre:
                scores_sorted, topk_inds = self.topKv2(
                    rpn_cls_score_process, self.topK_stage1[idx]
                )

                topk_inds = self.reshape(topk_inds, self.topK_shape[idx])

                bboxes_sorted = self.gatherND(rpn_bbox_pred_process, topk_inds)
                anchors_sorted = self.cast(
                    self.gatherND(anchors, topk_inds), self.ms_type
                )
                scores_sorted = self.reshape(
                    scores_sorted, self.topK_shape[idx]
                )
                anchors, scores, bboxes = (
                    anchors_sorted, scores_sorted, bboxes_sorted
                )
            else:
                scores = self.reshape(scores, (-1, 1))
            proposals_decode = self.bbox_coder.decode(
                anchors, bboxes, shape  # self.img_shape
            )

            proposals_decode = self.concat_axis1((proposals_decode, scores))
            proposals, _, mask_valid = self.nms(proposals_decode)
            mlvl_proposals = mlvl_proposals + (proposals,)
            mlvl_mask = mlvl_mask + (mask_valid,)

        proposals = self.concat_axis0(mlvl_proposals)
        masks = self.concat_axis0(mlvl_mask)

        x1, y1, x2, y2, scores = self.split(proposals)
        w = x2 - x1
        h = y2 - y1
        min_bbox_mask = ops.logical_and(
            ops.ge(w, self.min_bbox_size),
            ops.ge(h, self.min_bbox_size)
        ).reshape(-1)
        masks = ops.logical_and(masks, min_bbox_mask)
        scores = self.squeeze(scores)
        topk_mask = self.cast(self.topK_mask, self.ms_type)

        scores_using = self.select(masks, scores, topk_mask)

        _, topk_inds = self.topKv2(scores_using, self.max_num)

        topk_inds = self.reshape(topk_inds, self.topK_shape_stage2)
        proposals = self.gatherND(proposals, topk_inds)
        masks = self.gatherND(masks, topk_inds)
        return proposals, masks
