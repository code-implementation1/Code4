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
"""Random sampler."""
from typing import Tuple

import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore import ops


class RandomSampler(nn.Cell):
    """Random sampler.

    Args:
        num (int): Max number of valid samples.
        pos_fraction (float): Max fraction of positive samples.
        num_bboxes (int): Number of proposals.
        num_gts (int): Max number of gt bboxes
        add_gt_as_proposals (bool): Whether to add ground truth
            boxes as proposals. Defaults to True.
        neg_pos_ub (int): Upper bound number of negative and
            positive samples. Defaults to -1.
    """

    def __init__(
            self, num: int, pos_fraction: float, num_bboxes: int,
            num_gts: int, add_gt_as_proposals: bool = False,
            neg_pos_ub: int = -1
    ):
        """Init RandomSampler."""
        super(RandomSampler, self).__init__()
        self.dtype = np.float32
        self.ms_type = ms.float32

        self.num_gts = num_gts
        self.num_bboxes = num_bboxes
        self.add_gt_as_proposals = add_gt_as_proposals
        assert 0. < pos_fraction < 1.
        self.num_expected_pos = int(num * pos_fraction)
        self.num_expected_neg = num
        self.pos_neg_ub = neg_pos_ub

        self.add_gt_as_proposals_valid = ms.Tensor(
            np.full(self.num_gts, self.add_gt_as_proposals, dtype=np.int32)
        )
        self.label_inds = ms.Tensor(
            np.arange(1, self.num_gts + 1).astype(np.int32)
        )
        self.assigned_pos_ones = ms.Tensor(
            np.array(np.ones(self.num_expected_pos), dtype=np.int32)
        )
        self.gt_ignores = ms.Tensor(np.full(self.num_gts, -1, dtype=np.int32))
        self.bboxs_neg_mask = ms.Tensor(
            np.zeros((self.num_expected_neg, 4), dtype=np.float32)
        )
        self.labels_neg_mask = ms.Tensor(
            np.array(np.zeros(self.num_expected_neg), dtype=np.int32)
        )
        self.bboxes_neg_mask = ms.Tensor(
            np.zeros((self.num_expected_neg, 4), dtype=self.dtype)
        )
        self.range_pos_size = ms.Tensor(
            np.arange(self.num_expected_pos).astype(self.dtype)
        )
        self.range_neg_size = ms.Tensor(
            np.arange(self.num_expected_neg).astype(self.dtype)
        )
        self.check_neg_mask = ms.Tensor(
            np.array(
                np.ones(self.num_expected_neg - self.num_expected_pos),
                dtype=np.bool_
            )
        )
        self.reshape_shape_pos = (self.num_expected_pos, 1)
        self.reshape_shape_neg = (self.num_expected_neg, 1)

        self.random_choice_with_mask_pos = ops.RandomChoiceWithMask(
            self.num_expected_pos
        )
        self.random_choice_with_mask_neg = ops.RandomChoiceWithMask(
            self.num_expected_neg
        )

    def construct(
            self, assigned_gt_inds: ms.Tensor, overlaps: ms.Tensor,
            max_overlaps: ms.Tensor, assigned_labels: ms.Tensor,
            gt_bboxes_i: ms.Tensor, gt_labels_i: ms.Tensor,
            gt_valids: ms.Tensor, bboxes: ms.Tensor,
            valid_mask: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        """Sample positive and negative bboxes.

        Args:
            assigned_gt_inds (ms.Tensor): Assigning result. Shape is
                (self.num_bboxes, ). Tensor, that show what bbox correspond to
                some GT bbox.
            overlaps (ms.Tensor): IoU table between proposes and GT bboxes.
                Shape is (self.num_gts, self.num_bboxes).
            max_overlaps (ms.Tensor): Max IoUs for each anchor bbox. Shape is
                (self.num_bboxes, ).
            assigned_labels (ms.Tensor): Labels corresponded to assigning
                results. Shape is (self.num_bboxes, ).
            gt_bboxes_i (ms.Tensor): GT bboxes.
            gt_labels_i (ms.Tensor): GT labels.
            gt_valids (ms.Tensor): Mask that shows what GT bboxes are valid.
            bboxes (ms.Tensor): Proposed bboxes. Shape is (self.num_bboxes, 4).
            valid_mask (ms.Tensor): Mask that shows what proposed bboxes are
                valid. Shape is (self.num_bboxes, ).

        Returns:
            ms.Tensor: Indices of positive samples. Shape is
                (self.num_expected_pos, ).
            ms.Tensor: Mask that shows that positive sample is valid. Shape is
                (self.num_expected_pos, ).
            ms.Tensor: Indices of negative samples. Shape is
                (self.num_expected_neg, ).
            ms.Tensor: Mask that shows that negative sample is valid. Shape is
                (self.num_expected_pos, ).
        """
        # Get pos index
        pos_index, pos_valid_index = self.random_choice_with_mask_pos(
            ops.gt(assigned_gt_inds, 0)
        )

        pos_check_valid = ops.cast(
            ops.gt(assigned_gt_inds, 0), self.ms_type
        )
        pos_check_valid = ops.reduce_sum(pos_check_valid, -1)
        pos_valid_index = ops.less(self.range_pos_size, pos_check_valid)
        pos_index = pos_index * ops.reshape(
            ops.cast(pos_valid_index, ms.int32), (self.num_expected_pos, 1))

        pos_valid_index = ops.cast(pos_valid_index, ms.int32)
        pos_index = ops.reshape(pos_index, self.reshape_shape_pos)
        pos_valid_index = ops.reshape(
            pos_valid_index, self.reshape_shape_pos
        )
        pos_index = pos_index * pos_valid_index

        num_sampled_pos = ops.reduce_sum(
            ops.cast(pos_valid_index, self.ms_type),
        )
        num_neg = self.num_expected_neg - num_sampled_pos

        if self.pos_neg_ub >= 0:
            _pos = ops.maximum(1., num_sampled_pos)
            neg_upper_bound = ops.cast(self.pos_neg_ub * _pos, self.ms_type)
            if neg_upper_bound < num_neg:
                num_neg = neg_upper_bound

        # Get neg index
        neg_index, neg_valid_index = self.random_choice_with_mask_neg(
            ops.equal(assigned_gt_inds, 0)
        )

        sampled_neg_mask = ops.less(self.range_neg_size, num_neg)
        neg_valid_index = ops.logical_and(
            sampled_neg_mask, neg_valid_index
        )
        neg_index = ops.reshape(neg_index, self.reshape_shape_neg)

        neg_valid_index = ops.cast(neg_valid_index, ms.int32)
        neg_valid_index = ops.reshape(
            neg_valid_index, self.reshape_shape_neg
        )
        neg_index = neg_index * neg_valid_index

        return pos_index, pos_valid_index, neg_index, neg_valid_index
