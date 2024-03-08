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
"""Max IoU Assigner."""
from typing import Tuple

import mindspore as ms
from mindspore import nn
from mindspore import ops

import numpy as np


class MaxIOUAssigner(nn.Cell):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.
    - -1: not valid sample
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index of assigned gt

    Args:
        num_bboxes (int): Number of proposals.
        num_gts (int): Maximum number of GT bboxes.
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
            `min_pos_iou` is set to avoid assigning bboxes that have extremely
            small iou with GT as positive samples. It brings about 0.3 mAP
            improvements in 1x schedule but does not affect the performance of
            3x schedule.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not
            allowed in the second stage. Details are demonstrated in
            Step 4.
    """

    def __init__(
            self,
            num_bboxes: int,
            num_gts: int,
            neg_iou_thr: float,
            pos_iou_thr: float,
            min_pos_iou: float,
            match_low_quality: bool = True
    ):
        """Init MaxIOUAssigner."""
        super().__init__()
        self.dtype = np.float32
        self.ms_type = ms.float32
        self.num_bboxes = num_bboxes
        self.match_low_quality = match_low_quality

        self.neg_iou_thr = ms.Tensor(neg_iou_thr, self.ms_type)
        self.pos_iou_thr = ms.Tensor(pos_iou_thr, self.ms_type)
        self.min_pos_iou = ms.Tensor(min_pos_iou, self.ms_type)
        self.zero_thr = ms.Tensor(0.0, self.ms_type)

        self.assigned_labels = ms.Tensor(
            np.full(num_bboxes, -1, dtype=np.int32)
        )
        self.assigned_gt_inds = ms.Tensor(
            np.full(num_bboxes, -1, dtype=np.int32)
        )
        self.assigned_gt_zeros = ms.Tensor(
            np.array(np.zeros(num_bboxes), dtype=np.int32)
        )
        self.assigned_gt_ignores = ms.Tensor(
            np.full(num_bboxes, -1, dtype=np.int32)
        )
        self.assigned_gt_ones = ops.Tensor(
            np.array(np.ones(num_bboxes), dtype=np.int32)
        )

        self.num_gts = num_gts
        self.check_gt_one = ms.Tensor(
            np.full((self.num_gts, 4), -1, dtype=self.dtype)
        )
        self.check_anchor_two = ms.Tensor(
            np.full((self.num_bboxes, 4), -2, dtype=self.dtype)
        )

        self.max_gt = ops.ArgMaxWithValue(axis=0)
        self.max_anchor = ops.ArgMaxWithValue(axis=1)
        self.greaterequal = ops.GreaterEqual()

    def construct(
            self, gt_bboxes_i: ms.Tensor, gt_labels_i: ms.Tensor,
            valid_mask: ms.Tensor, bboxes: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        """Assign bounding bboxes.

        Args:
            gt_bboxes_i: GT bboxes (self.num_gts, 4).
            gt_labels_i: GT labels (self.num_gts,).
            valid_mask: Mask that shows valid proposed bboxes
                (self.num_bboxes, ).
            bboxes: Proposed bboxes (self.num_bboxes, 4).


        Returns:
            ms.Tensor: Assigning results (self.num_bboxes, ). Tensor, that
                show what bbox correspond to some GT bbox.
            ms.Tensor: IoU table between proposes and GT bboxes.
                Shape is (self.num_gts, self.num_bboxes).
            ms.Tensor: Max IoUs for each anchor bbox. Shape is
                (self.num_bboxes, ).
            ms.Tensor: Corresponded labels for assigning result. Shape is
                (self.num_bboxes, ).
        """
        overlaps = bbox_overlaps(bboxes, gt_bboxes_i)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        argmax_overlaps, max_overlaps = self.max_gt(overlaps)
        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        _, max_overlaps_gt = self.max_anchor(overlaps)
        # 2. assign negative: below
        # the negative inds are set to be 0
        neg_sample_iou_mask = ops.logical_and(
            ops.ge(max_overlaps, self.zero_thr),
            ops.less(max_overlaps, self.neg_iou_thr)
        )
        assigned_gt_inds2 = ops.select(
            neg_sample_iou_mask, self.assigned_gt_zeros, self.assigned_gt_inds
        )
        # 3. assign positive: above positive IoU threshold
        pos_sample_iou_mask = self.greaterequal(
            max_overlaps, self.pos_iou_thr
        )
        assigned_gt_inds3 = ops.select(
            pos_sample_iou_mask,
            argmax_overlaps + self.assigned_gt_ones,
            assigned_gt_inds2
        )

        assigned_gt_inds4 = assigned_gt_inds3
        if self.match_low_quality:
            # Low-quality matching will overwrite the assigned_gt_inds assigned
            # in Step 3. Thus, the assigned gt might not be the best one for
            # prediction.
            # For example, if bbox A has 0.9 and 0.8 iou with GT bbox 1 & 2,
            # bbox 1 will be assigned as the best target for bbox A in step 3.
            # However, if GT bbox 2's gt_argmax_overlaps = A, bbox A's
            # assigned_gt_inds will be overwritten to be bbox 2.
            # This might be the reason that it is not used in ROI Heads.
            for j in range(self.num_gts):
                max_overlaps_gt_j = max_overlaps_gt[j:j + 1:1]
                overlaps_gt_j = ops.squeeze(overlaps[j:j + 1:1, ::])

                pos_mask_j = ops.logical_and(
                    ops.ge(max_overlaps_gt_j, self.min_pos_iou),
                    ops.equal(overlaps_gt_j, max_overlaps_gt_j)
                )

                assigned_gt_inds4 = ops.select(
                    pos_mask_j, self.assigned_gt_ones + j, assigned_gt_inds4
                )

        assigned_gt_inds5 = ops.select(
            valid_mask, assigned_gt_inds4, self.assigned_gt_ignores
        )

        if gt_labels_i is not None:
            pos_mask = assigned_gt_inds5 > 0
            temp_gt_inds = ops.select(
                pos_mask, assigned_gt_inds5 - 1, self.assigned_gt_zeros
            )
            temp_labels = gt_labels_i[temp_gt_inds]
            assigned_labels = ops.select(
                pos_mask, ops.cast(temp_labels, ms.int32),
                self.assigned_labels
            )
        else:
            assigned_labels = self.assigned_labels

        return assigned_gt_inds5, overlaps, max_overlaps, assigned_labels

def bbox_overlaps(bboxes1, bboxes2, eps=1e-6):
    """Calculate overlap between two set of bboxes."""
    area1 = (bboxes1[::, 2] - bboxes1[::, 0]) * (bboxes1[::, 3] - bboxes1[::, 1])
    area2 = (bboxes2[::, 2] - bboxes2[::, 0]) * (bboxes2[::, 3] - bboxes2[::, 1])

    lt = ms.ops.maximum(bboxes1[:, None, :2], bboxes2[None, :, :2])
    rb = ms.ops.minimum(bboxes1[:, None, 2:], bboxes2[None, :, 2:])

    wh = ops.clip(rb - lt, min=0)
    overlap = wh[..., 0] * wh[..., 1]
    union = area1[..., None] + area2[None, :] - overlap
    ious = overlap / (union + eps)
    ious = ops.t(ious)
    return ious