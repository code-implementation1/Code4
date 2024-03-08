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
"""Functional to build assigner sampler object."""
from typing import Sequence

from .sampling import AssignerSampler
from .sampling import MaxIOUAssigner
from .sampling import RandomSampler
from .sampling import BalancedSampler

from .bbox_coder import DeltaXYWHBBoxCoder


def build_max_iou_random(
        num_bboxes: int,
        num_gts: int,
        pos_num_expected: int,
        neg_num_expected: int,
        neg_iou_thr: float = 0.3,
        pos_iou_thr: float = 0.7,
        min_pos_iou: float = 0.3,
        add_gt_as_proposals: bool = False,
        match_low_quality: bool = True,
        neg_pos_ub: int = -1,
        target_means: Sequence[float] = (0., 0., 0., 0.),
        target_stds: Sequence[float] = (1., 1., 1., 1.),
        rcnn_mode: bool = False
) -> AssignerSampler:
    """Build MaxIOUAssgner and RandomSampler."""
    assigner = MaxIOUAssigner(
        num_bboxes=num_bboxes,
        num_gts=num_gts,
        neg_iou_thr=neg_iou_thr,
        pos_iou_thr=pos_iou_thr,
        min_pos_iou=min_pos_iou,
        match_low_quality=match_low_quality
    )

    random_sampler = RandomSampler(
        pos_expected=pos_num_expected, neg_expected=neg_num_expected,
        num_bboxes=num_bboxes, num_gts=num_gts,
        add_gt_as_proposals=add_gt_as_proposals,
        neg_pos_ub=neg_pos_ub
    )
    bbox_coder = DeltaXYWHBBoxCoder(
        target_stds=target_stds, target_means=target_means
    )
    assigner_sampler = AssignerSampler(
        assigner=assigner, sampler=random_sampler, rcnn_mode=rcnn_mode,
        bbox_coder=bbox_coder
    )
    return assigner_sampler


def build_max_iou_balanced(
        num_bboxes: int,
        num_gts: int,
        pos_num_expected: int,
        neg_num_expected: int,
        neg_iou_thr: float = 0.3,
        pos_iou_thr: float = 0.7,
        min_pos_iou: float = 0.3,
        add_gt_as_proposals: bool = False,
        match_low_quality: bool = True,
        floor_thr: int = -1,
        floor_fraction: int = 0,
        num_bins: int = 3,
        neg_pos_ub: int = -1,
        target_means: Sequence[float] = (0., 0., 0., 0.),
        target_stds: Sequence[float] = (1., 1., 1., 1.),
        rcnn_mode: bool = False
) -> AssignerSampler:
    """Build MaxIOUAssigner and BalancedSampler."""
    assigner = MaxIOUAssigner(
        num_bboxes=num_bboxes,
        num_gts=num_gts,
        neg_iou_thr=neg_iou_thr,
        pos_iou_thr=pos_iou_thr,
        min_pos_iou=min_pos_iou,
        match_low_quality=match_low_quality
    )
    balanced_sampler = BalancedSampler(
        pos_expected=pos_num_expected, neg_expected=neg_num_expected,
        num_bboxes=num_bboxes, num_gts=num_gts,
        add_gt_as_proposals=add_gt_as_proposals,
        floor_thr=floor_thr,
        floor_fraction=floor_fraction,
        num_bins=num_bins,
        neg_pos_ub=neg_pos_ub
    )
    bbox_coder = DeltaXYWHBBoxCoder(target_stds=target_stds,
                                    target_means=target_means)
    balanced_assigner_sampler = AssignerSampler(
        assigner=assigner, sampler=balanced_sampler, rcnn_mode=rcnn_mode,
        bbox_coder=bbox_coder
    )
    return balanced_assigner_sampler
