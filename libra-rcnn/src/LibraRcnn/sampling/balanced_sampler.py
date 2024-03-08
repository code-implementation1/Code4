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
# This file has been derived from the https://github.com/open-mmlab/mmdetection/tree/v2.28.2
# repository and modified.
# ============================================================================
"""Balanced sampler."""
from typing import Tuple

import numpy as np

import mindspore as ms
from mindspore import nn
from mindspore import ops


class BalancedSampler(nn.Cell):
    """Balanced sampler.

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)

    Combined sampler that call InstanceBalancedPosSampler
    and IoUBalancedNegSampler.

    Args:
        pos_expected (int): Max number of positive samples.
        neg_expected (int): Max number of negative samples.
        num_bboxes (int): Number of proposals.
        num_gts: Max number of gt bboxes.
        floor_thr (float): threshold (minimum) IoU for IoU balanced sampling,
            set to -1 if all using IoU balanced sampling.
        floor_fraction (float): sampling fraction of proposals under floor_thr.
        num_bins (int): number of bins in IoU balanced sampling.
        add_gt_as_proposals (bool): Whether to add ground truth
            boxes as proposals. Defaults to True.
        neg_pos_ub (int): Upper bound number of negative and
            positive samples. Defaults to -1.
    """

    def __init__(
            self,
            pos_expected: int,
            neg_expected: int,
            num_bboxes: int,
            num_gts: int,
            floor_thr: float = -1,
            floor_fraction: float = 0,
            num_bins: int = 3,
            add_gt_as_proposals: bool = False,
            neg_pos_ub: int = -1
    ):
        """Init BalancedSampler."""
        super(BalancedSampler, self).__init__()
        self.dtype = np.float32
        self.ms_type = ms.float32

        self.num_gts = num_gts
        self.num_bboxes = num_bboxes
        self.add_gt_as_proposals = add_gt_as_proposals
        self.num_expected_pos = pos_expected
        self.num_expected_neg = neg_expected
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
        self.bboxes_neg_mask = ms.Tensor(
            np.zeros((self.num_expected_neg, 4), dtype=np.float32)
        )
        self.labels_neg_mask = ms.Tensor(
            np.array(np.zeros(self.num_expected_neg), dtype=np.int32)
        )
        self.reshape_shape_pos = (self.num_expected_pos, 1)
        self.reshape_shape_neg = (self.num_expected_neg, 1)

        self.pos_sampler = InstanceBalancedPosSampler(
            num_expected=pos_expected, num_gts=num_gts,
            num_bboxes=(
                num_bboxes + num_gts
                if self.add_gt_as_proposals else num_bboxes
            )
        )
        self.neg_sampler = IoUBalancedNegSampler(
            num_bboxes=(
                num_bboxes + num_gts
                if self.add_gt_as_proposals else num_bboxes
            ),
            neg_num_expected=neg_expected,
            floor_thr=floor_thr, floor_fraction=floor_fraction,
            num_bins=num_bins
        )

    def construct(
            self, assigned_gt_inds: ms.Tensor, overlaps: ms.Tensor,
            max_overlaps: ms.Tensor, assigned_labels: ms.Tensor,
            gt_bboxes_i: ms.Tensor, gt_labels_i: ms.Tensor,
            valid_mask: ms.Tensor, bboxes: ms.Tensor, gt_valids: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        """Sample positive and negative bboxes.

        Args:
            assigned_gt_inds (ms.Tensor): Assigning result.
            overlaps (ms.Tensor): IoUs between GT and proposed boxes
            max_overlaps (ms.Tensor): Max IoUs between GT and proposed boxes.
            assigned_labels (ms.Tensor): Labels corresponded to assigning
                results.
            gt_bboxes_i (ms.Tensor): GT bboxes.
            gt_labels_i (ms.Tensor): GT labels
            valid_mask (ms.Tensor): Mask that shows valid proposed bboxes.
            bboxes (ms.Tensor): Proposed bboxes.
            gt_valids (ms.Tensor): Mask that shows valid GT bboxes.

        Returns:
            ms.Tensor: Indices of positive samples.
            ms.Tensor: Mask that shows that positive sample is valid.
            ms.Tensor: Indices of negative samples.
            ms.Tensor: Mask that shows that negative sample is valid.
        """
        # pos indices
        pos_index, pos_valid_index = self.pos_sampler(
            assigned_gt_inds, gt_valids
        )
        # define the number of rest negative samples.
        pos_num = ops.reduce_sum(
            ops.cast(pos_valid_index, ms.float32)
        )
        num_neg = self.num_expected_neg - pos_num
        # if pos_neg_ub >= 0, change num_neg so that
        # `num_neg / pos_num <= pos_neg_ub` is true.
        if self.pos_neg_ub >= 0:
            _pos = ops.maximum(1., pos_num)
            neg_upper_bound = ops.cast(self.pos_neg_ub * _pos, self.ms_type)
            if neg_upper_bound < num_neg:
                num_neg = neg_upper_bound

        # neg indices
        neg_index, neg_valid_index = self.neg_sampler(
            assigned_gt_inds, max_overlaps, num_neg
        )

        return pos_index, pos_valid_index, neg_index, neg_valid_index


class InstanceBalancedPosSampler(nn.Cell):
    """Instance balanced sampler that samples equal number of positive samples
    for each instance."""

    def __init__(self, num_expected: int, num_gts: int, num_bboxes: int):
        """Init InstanceBalancedPosSampler."""
        super(InstanceBalancedPosSampler, self).__init__()
        self.dtype = np.float32
        self.ms_type = ms.float32

        self.num_expected = num_expected
        self.num_gts = num_gts
        self.num_bboxes = num_bboxes
        self.reshape_shape_pos = (self.num_expected, 1)
        self.random_choice_with_mask = ops.RandomChoiceWithMask(
            self.num_expected
        )
        self.pos_range = ms.Tensor(
            np.arange(self.num_expected, dtype=np.float32)
        )

        self.assigned_pos_ones = ms.Tensor(
            np.array(np.ones(self.num_expected), dtype=np.int32)
        )

    def construct(
            self, assigned_gt_inds: ms.Tensor, gt_valids: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """Sample positive samples.

        Args:
            assigned_gt_inds (ms.Tensor): Tensor with length equal the
                number of proposals. Shows indices of corresponded gt bboxes.
            gt_valids (ms.Tensor): Mask than shows valid gt bboxes.

        Returns:
            ms.Tensor: sampled indices
            ms.Tensor: sampled indices mask
        """
        pos_mask = ops.gt(assigned_gt_inds, 0)
        ins_pos_inds, ins_pos_mask = self.sample_by_instances(
            assigned_gt_inds, gt_valids
        )

        chosen_num = ops.reduce_sum(ops.cast(ins_pos_inds, ms.float32))

        num_extra = self.num_expected - chosen_num
        mask_extra = ops.scatter_nd(
            ins_pos_inds,
            ops.cast(ops.reshape(ins_pos_mask, (-1, 1)), ms.int32),
            (self.num_bboxes, 1)
        )
        mask_extra = ops.reshape(
            ops.logical_not(ops.cast(mask_extra, ms.bool_)),
            (self.num_bboxes,)
        )

        mask_extra = ops.logical_and(mask_extra, pos_mask)
        extra_inds, extra_mask = self.sample_rest_num(num_extra, mask_extra)

        inds_total = ops.concat((ins_pos_inds, extra_inds), axis=0)
        mask_total = ops.concat((ins_pos_mask, extra_mask), axis=0)

        chosen_inds, chosen_mask = self.random_choice_with_mask(mask_total)
        pos_index = ops.gather_nd(inds_total, chosen_inds)
        valid_pos_index = ops.gather_nd(mask_total, chosen_inds)
        valid_pos_index = ops.logical_and(valid_pos_index, chosen_mask)

        pos_index = ops.reshape(pos_index, self.reshape_shape_pos)

        valid_pos_index = ops.cast(valid_pos_index, ms.int32)
        valid_pos_index = ops.reshape(
            valid_pos_index, self.reshape_shape_pos
        )
        pos_index = pos_index * valid_pos_index
        return pos_index, valid_pos_index

    def sample_by_instances(
            self, assigned_gt_inds: ms.Tensor, gt_valids: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """Sample positive boxes per instance gt bbox.

        Args:
            assigned_gt_inds (ms.Tensor): Result of assigning.
            gt_valids (ms.Tensor): Mask that show that GT bbox is valid.

        Returns:
            ms.Tensor: Positive samples indices.
            ms.Tensor: Mask of valid positive indices.
        """
        gt_num = ops.reduce_sum(ops.cast(gt_valids, ms.float32), -1)
        num_per_gt = ops.cast(self.num_expected / gt_num + 1, ms.int64)
        inds_list = []
        mask_list = []

        max_mask = self.pos_range < num_per_gt
        # collect indices for each potential gt bbox
        for i in range(self.num_gts):
            inds, mask = self.random_choice_with_mask(
                ops.equal(assigned_gt_inds, i + 1)
            )

            mask = ops.logical_and(mask, max_mask)
            inds_list.append(inds)
            mask_list.append(mask)

        # unite sampled bboxes indices.
        inds_total = ops.concat(inds_list, axis=0)
        mask_total = ops.concat(mask_list, axis=0)

        chosen_inds, chosen_mask = self.random_choice_with_mask(mask_total)

        pos_index = ops.gather_nd(inds_total, chosen_inds)
        valid_pos_index = ops.gather_nd(mask_total, chosen_inds)
        valid_pos_index = ops.logical_and(valid_pos_index, chosen_mask)

        return pos_index, valid_pos_index

    def sample_rest_num(
            self, num: int, mask: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """Sample bboxes by mask and required number."""
        expected_mask = ops.less(
            self.pos_range, ops.cast(num, ms.float32)
        )
        sampled_inds, sampled_mask = self.random_choice_with_mask(mask)
        sampled_mask = ops.logical_and(sampled_mask, expected_mask)
        return sampled_inds, sampled_mask


class IoUBalancedNegSampler(nn.Cell):
    """IoU Balanced Sampling.

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)

    Sampling proposals according to their IoU. `floor_fraction` of needed
    RoIs are sampled from proposals whose IoU are lower than `floor_thr`
    randomly. The others are sampled from proposals whose IoU are higher
    than `floor_thr`. These proposals are sampled from some bins evenly,
    which are split by `num_bins` via IoU evenly.

    Args:
        num_bboxes (int): number of proposals.
        neg_num_expected (float): maximal number of negative samples.
        floor_thr (float): threshold (minimum) IoU for IoU balanced sampling,
            set to -1 if all using IoU balanced sampling.
        floor_fraction (float): sampling fraction of proposals under floor_thr.
        num_bins (int): number of bins in IoU balanced sampling.
    """

    def __init__(
            self, num_bboxes: int, neg_num_expected: int, floor_thr: float = 0,
            floor_fraction: float = 0, num_bins: int = 3
    ):
        """Init IoUBalancedNegSampler."""
        super(IoUBalancedNegSampler, self).__init__()
        self.num_bboxes = num_bboxes
        self.num_expected = neg_num_expected
        self.floor_thr = floor_thr
        self.floor_fraction = floor_fraction
        self.num_bins = num_bins
        self.empty_mask = ms.Tensor(np.zeros(num_bboxes, dtype=np.bool_))

        self.neg_range = ms.Tensor(
            np.arange(self.num_expected, dtype=np.float32)
        )

        self.random_choice_with_mask = ops.RandomChoiceWithMask(
            self.num_expected
        )
        self.reshape_shape_neg = (self.num_expected, 1)

    def construct(
            self, assigned_gt_inds: ms.Tensor, max_overlaps: ms.Tensor,
            num_neg: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """Sample negative samples.

        Args:
            assigned_gt_inds (ms.Tensor): Tensor with length equal the
                number of proposals. Shows indices of corresponded gt bboxes.
            max_overlaps (ms.Tensor): Tensor with length equal the
                number of proposals. Shows IoU between proposes bbox and
                corresponded gt bbox.
            num_neg (Union[ms.Tensor, float]): Number of expected
                negative samples (<=self.num_expected).

        Returns:
            ms.Tensor: sampled indices
            ms.Tensor: sampled indices mask
        """
        # define iou sampled bboxes and bboxes with iou less than threshold
        neg_mask = ops.equal(assigned_gt_inds, 0)
        floor_thr = self.floor_thr
        if floor_thr > 0:
            floor_mask = ops.logical_and(
                ops.ge(max_overlaps, 0),
                ops.less(max_overlaps, floor_thr)
            )
            iou_sampling_mask = ops.ge(max_overlaps, floor_thr)
        elif floor_thr == 0:
            floor_mask = ops.equal(max_overlaps, 0)
            iou_sampling_mask = ops.ge(max_overlaps, floor_thr)
        else:
            floor_mask = self.empty_mask
            iou_sampling_mask = ops.gt(max_overlaps, floor_thr)
            floor_thr = 0
        floor_neg_mask = ops.logical_and(floor_mask, neg_mask)
        iou_sampling_neg_mask = ops.logical_and(iou_sampling_mask, neg_mask)
        num_expected_iou_sampling = ops.cast(
            num_neg * (1 - self.floor_fraction), ms.int32
        )
        # sample iou sampled bboxes
        iou_sampled_inds, iou_sampled_mask = self.sample_via_interval(
            max_overlaps, iou_sampling_neg_mask, num_expected_iou_sampling,
            floor_thr
        )
        num_iou_sampled = ops.cast(
            ops.reduce_sum(ops.cast(iou_sampled_mask, ms.float32)),
            ms.int64
        )
        # sample bboxes with gt less than threshold
        num_expected_floor = num_neg - num_iou_sampled
        sampled_floor_inds, sampled_floor_mask = self.sample_rest_num(
            num_expected_floor, floor_neg_mask
        )
        num_floor_sampled = ops.cast(
            ops.reduce_sum(ops.cast(sampled_floor_mask, ms.float32)),
            ms.int64
        )
        # sample extra bboxes
        num_extra = num_neg - num_iou_sampled - num_floor_sampled
        mask_extra1 = ops.scatter_nd(
            sampled_floor_inds,
            ops.cast(ops.reshape(sampled_floor_mask, (-1, 1)), ms.int32),
            (self.num_bboxes, 1)
        )
        mask_extra2 = ops.scatter_nd(
            iou_sampled_inds,
            ops.cast(ops.reshape(iou_sampled_mask, (-1, 1)), ms.int32),
            (self.num_bboxes, 1)
        )
        mask_extra = ops.reshape(
            ops.logical_not(
                ops.cast(mask_extra1 + mask_extra2, ms.bool_)
            ),
            (self.num_bboxes,)
        )
        mask_extra = ops.logical_and(mask_extra, neg_mask)
        extra_inds, extra_mask = self.sample_rest_num(num_extra, mask_extra)

        # unite all sampled bboxes
        inds_total = ops.concat(
            (iou_sampled_inds, sampled_floor_inds, extra_inds), axis=0
        )
        mask_total = ops.concat(
            (iou_sampled_mask, sampled_floor_mask, extra_mask), axis=0
        )
        chosen_inds, chosen_mask = self.random_choice_with_mask(mask_total)
        neg_index = ops.gather_nd(inds_total, chosen_inds)
        valid_neg_index = ops.gather_nd(mask_total, chosen_inds)
        valid_neg_index = ops.logical_and(
            valid_neg_index, chosen_mask
        )

        sampled_neg_mask = ops.less(self.neg_range, num_neg)
        valid_neg_index = ops.logical_and(
            sampled_neg_mask, valid_neg_index
        )
        neg_index = ops.reshape(neg_index, self.reshape_shape_neg)

        valid_neg_index = ops.cast(valid_neg_index, ms.int32)
        valid_neg_index = ops.reshape(
            valid_neg_index, self.reshape_shape_neg
        )
        neg_index = neg_index * valid_neg_index

        return neg_index, valid_neg_index

    def sample_rest_num(
            self, num: int, mask: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """Sample bboxes by mask and required number."""
        expected_mask = ops.less(
            self.neg_range, ops.cast(num, ms.float32)
        )
        sampled_inds, sampled_mask = self.random_choice_with_mask(mask)
        sampled_mask = ops.logical_and(sampled_mask, expected_mask)
        return sampled_inds, sampled_mask

    def sample_via_interval(
            self, max_overlaps: ms.Tensor, iou_sampling_neg_mask: ms.Tensor,
            num_expected: int, floor_thr: float
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """Sample according to the iou interval.

        Args:
            max_overlaps (ms.Tensor): IoU between bounding boxes and
                ground truth boxes.
            iou_sampling_neg_mask (ms.Tensor): Mask of iou_sampled
                tensors.
            num_expected (int): Number of expected samples.
            floor_thr (float): Low IOU border for iou sampled bboxes.

        Returns:
            ms.Tensor: Indices  of samples
        """
        max_iou = max_overlaps.max()
        iou_interval = (max_iou - floor_thr) / self.num_bins
        per_num_expected = ops.cast(num_expected / self.num_bins, ms.int64)

        max_mask = ops.less(self.neg_range, per_num_expected)

        inds_list = []
        mask_list = []
        # collect bboxes for each interval
        for i in range(self.num_bins):
            start_iou = floor_thr + i * iou_interval
            end_iou = floor_thr + (i + 1) * iou_interval
            tmp_mask = ops.logical_and(
                ops.ge(max_overlaps, start_iou),
                ops.less(max_overlaps, end_iou)
            )
            tmp_mask = ops.logical_and(tmp_mask, iou_sampling_neg_mask)
            inds, mask = self.random_choice_with_mask(tmp_mask)
            mask = ops.logical_and(mask, max_mask)
            inds_list.append(inds)
            mask_list.append(mask)

        # unite sampled bboxes indices
        inds_total = ops.concat(inds_list, axis=0)
        mask_total = ops.concat(mask_list, axis=0)

        chosen_inds, chosen_mask = self.random_choice_with_mask(mask_total)

        sampled_index = ops.gather_nd(inds_total, chosen_inds)
        valid_sampled_index = ops.gather_nd(mask_total, chosen_inds)
        valid_sampled_index = ops.logical_and(
            valid_sampled_index, chosen_mask
        )
        return sampled_index, valid_sampled_index
