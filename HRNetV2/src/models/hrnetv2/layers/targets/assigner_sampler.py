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
"""Wrapper for assigner and sampler that generate targets."""
from typing import Any, Tuple

import mindspore as ms
from mindspore import nn
from mindspore import ops


class AssignerSampler(nn.Cell):
    """Wrapper for pair assigner-sampler for MaskRCNN models.
    Prepares bboxes, calls assigner, add additional gt proposals (if
    necessary), calls sampler, prepares results (with segmentation masks).

    Args:
        assigner: Used bounding boxes assigner.
        sampler: Used bounding boxes sampler.
        bbox_coder: Used bbox coder to generate target values by sampled
            and gt samples.
        rcnn_mode (bool): Whether return results in RCNN (sampled bboxes,
            localization targets, classification targets, classification mask)
            or RPN (localization targets, localization mask, classification
            targets, classification mask) format.
    """

    def __init__(
            self, assigner: Any, sampler: Any, bbox_coder: Any,
            rcnn_mode: bool = True
    ):
        """Init AssignerSampler."""
        super().__init__()
        self.assigner = assigner
        self.sampler = sampler
        self.rcnn_mode = rcnn_mode
        self.bbox_coder = bbox_coder

    def prepare(
            self, gt_bboxes_i: ms.Tensor, gt_valids: ms.Tensor,
            bboxes: ms.Tensor, valid_mask: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """Prepare gt_bboxes_i and bboxes before assignment.

        Args:
            gt_bboxes_i (ms.Tensor): GT bboxes for current image in batch.
            gt_valids (ms.Tensor): Mask that shows which GT bboxes are valid.
            bboxes (ms.Tensor): Proposed bboxes.
            valid_mask (ms.Tensor): Mask that shows which boxes ara valid and
                can be used in assigning and sampling.

        Returns:
            ms.Tensor: Prepared GT bboxes.
            ms.Tensor: Prepared proposed bboxes.
        """
        gt_bboxes_i_mask = ops.cast(
            ops.tile(
                ops.reshape(
                    ops.cast(gt_valids, ms.int32),
                    (self.assigner.num_gts, 1)
                ),
                (1, 4)
            ),
            ms.bool_
        )
        gt_bboxes_i = ops.select(
            gt_bboxes_i_mask, gt_bboxes_i, self.assigner.check_gt_one
        )
        bboxes_mask = ops.cast(
            ops.tile(
                ops.reshape(
                    ops.cast(valid_mask, ms.int32),
                    (self.assigner.num_bboxes, 1)
                ),
                (1, 4)
            ),
            ms.bool_
        )
        bboxes = ops.select(
            bboxes_mask, bboxes, self.assigner.check_anchor_two
        )

        return gt_bboxes_i, bboxes

    def get_result(
            self, assigned_gt_inds: ms.Tensor, bboxes: ms.Tensor,
            gt_labels_i: ms.Tensor, gt_bboxes_i: ms.Tensor,
            pos_index: ms.Tensor, pos_valid_index: ms.Tensor,
            neg_index: ms.Tensor, neg_valid_index: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        """Prepare result of sampling as targets and weights.

        Args:
            assigned_gt_inds (ms.Tensor): Assigning result.
            bboxes (ms.Tensor): Proposed bboxes.
            gt_labels_i (ms.Tensor): GT labels for current image.
            gt_bboxes_i (ms.Tensor): GT bboxes for current image.
            pos_index (ms.Tensor): List of positive indices.
            pos_valid_index (ms.Tensor): Mask that shows valid positive
                indices in the list
            neg_index (ms.Tensor): List of negative indices.
            neg_valid_index (ms.Tensor): Mask that shows valid negative
                indices in the list.

        Returns:
             ms.Tensor: Sampled bboxes (RCNN) or localization targets (RPN)
             ms.Tensor: Localization targets (RCNN) or localization mask (RPN)
             ms.Tensor: Classification targets
             ms.Tensor: Classification mask
        """
        pos_assigned_gt_index = ops.gather_nd(
            assigned_gt_inds, pos_index
        ) - self.sampler.assigned_pos_ones
        pos_assigned_gt_index = ops.reshape(
            pos_assigned_gt_index, self.sampler.reshape_shape_pos
        )
        pos_assigned_gt_index = pos_assigned_gt_index * pos_valid_index
        pos_gt_labels = ops.gather_nd(gt_labels_i, pos_assigned_gt_index)
        pos_bboxes_ = ops.gather_nd(bboxes, pos_index)
        neg_bboxes_ = ops.gather_nd(bboxes, neg_index)

        pos_assigned_gt_index = ops.reshape(
            pos_assigned_gt_index, self.sampler.reshape_shape_pos
        )
        pos_gt_bboxes_ = ops.gather_nd(gt_bboxes_i, pos_assigned_gt_index)

        pos_bbox_targets_ = self.bbox_coder.encode(pos_bboxes_, pos_gt_bboxes_)

        if self.rcnn_mode:
            # Prepare targets of RCNN:
            # total_bboxes (rois),
            # total_deltas (corresponded gt bboxes deltas),
            # total_labels (corresponded gt labels (categories ids)),
            # total_mask (corresponded mask (show that bbox is valid))
            total_bboxes = ops.concat((pos_bboxes_, neg_bboxes_))
            total_deltas = ops.concat(
                (pos_bbox_targets_, self.sampler.bboxes_neg_mask))
            total_labels = ops.concat(
                (ops.cast(pos_gt_labels, ms.int32),
                 self.sampler.labels_neg_mask)
            )

            valid_pos_index = ops.reshape(
                pos_valid_index, self.sampler.reshape_shape_pos
            )
            valid_neg_index = ops.reshape(
                neg_valid_index, self.sampler.reshape_shape_neg
            )
            total_mask = ops.concat((valid_pos_index, valid_neg_index))
            output = (
                total_bboxes, total_deltas, total_labels, total_mask,
                pos_bboxes_, pos_assigned_gt_index, pos_gt_labels, valid_pos_index
            )
        else:
            # Prepare targets of RPN:
            # bbox_targets_total (gt bboxes deltas)
            # bbox_weights_total (show that bbox is valid)
            # labels_total (gt labels)
            # label_weights_total (show that label is valid)
            valid_pos_index = ops.cast(pos_valid_index, ms.int32)
            valid_neg_index = ops.cast(neg_valid_index, ms.int32)

            valid_pos_index = ops.reshape(
                valid_pos_index, (self.sampler.num_expected_pos,)
            )
            valid_neg_index = ops.reshape(
                valid_neg_index, (self.sampler.num_expected_neg,)
            )
            bbox_targets_total = ops.scatter_nd(
                pos_index, pos_bbox_targets_,
                (self.sampler.num_bboxes, 4)
            )

            bbox_weights_total = ops.scatter_nd(
                pos_index, valid_pos_index,
                (self.sampler.num_bboxes,)
            )

            total_index = ops.concat((pos_index, neg_index))
            total_valid_index = ops.concat(
                (valid_pos_index, valid_neg_index)
            )
            label_weights_total = ops.scatter_nd(
                total_index, total_valid_index,
                (self.sampler.num_bboxes,)
            )
            labels_total = ops.scatter_nd(
                pos_index, pos_gt_labels,
                (self.sampler.num_bboxes,)
            )

            output = (
                bbox_targets_total, ops.cast(bbox_weights_total, ms.bool_),
                labels_total, ops.cast(label_weights_total, ms.bool_)
            )
        return output

    def apply_gt_proposals(
            self, bboxes: ms.Tensor, gt_bboxes_i: ms.Tensor,
            gt_valids: ms.Tensor, assigned_gt_inds: ms.Tensor,
            max_overlaps: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        """Add gt bbox as proposals to assigned bboxes.

        Args:
            bboxes (ms.Tensor): Proposed bboxes or anchors.
            gt_bboxes_i (ms.Tensor): GT bounding boxes.
            gt_valids (ms.Tensor): Mask that shows valid GT bboxes.
            assigned_gt_inds (ms.Tensor): Assigning results (labels of bboxes)
            max_overlaps (ms.Tensor): Assigning results (max IOUs between gt
                and proposals).

        Returns:
            Tuple[ms.Tensor, ms.Tensor, ms.Tensor]: `bboxes`,
                `assigned_gt_inds` and `max_overlaps` with added GT info.
        """
        bboxes = ops.concat((gt_bboxes_i, bboxes))
        label_inds_valid = ops.select(
            gt_valids, self.sampler.label_inds, self.sampler.gt_ignores
        )
        label_inds_valid = (
            label_inds_valid * self.sampler.add_gt_as_proposals_valid
        )
        assigned_gt_inds = ops.concat((label_inds_valid, assigned_gt_inds))

        gt_overlaps = ops.cast(
            ops.cast(
                gt_valids, ms.int32
            ) * self.sampler.add_gt_as_proposals_valid,
            ms.float32
        )
        max_overlaps = ops.concat((gt_overlaps, max_overlaps))

        return bboxes, assigned_gt_inds, max_overlaps

    def construct(
            self, gt_bboxes_i: ms.Tensor,
            gt_labels_i: ms.Tensor, bboxes: ms.Tensor, gt_valids: ms.Tensor,
            valid_mask: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        """Assign and sample bboxes.

        Args:
            gt_bboxes_i (ms.Tensor): GT bboxes. Tensor has a fixed shape
                (n, 4), where n equals to self.assigner.num_gts.
            gt_labels_i (ms.Tensor): GT labels. Tensor has a fixed shape (n,),
                where n equals to self.assigner.num_gts.
            bboxes (ms.Tensor): Proposed bboxes or anchors. Tensor has a fixed
                shape (n, 4), where n is self.assigner.num_bboxes.
            gt_valids (ms.Tensor): Mask that shows valid GT bboxes. Tensor has
                a fixed shape (n,), where n equals to self.assigner.num_gts.
            valid_mask (ms.Tensor): Mask that shows valid proposed bboxes.
                Tensor has a fixed shape (n, 4), where n is
                self.assigner.num_bboxes.

        Returns:
            ms.Tensor: Sampled bboxes with shape
                (self.sampler.num_expected_neg + self.sampler.num_expected_pos, 4)
                if self.rcnn_mode else or localization target deltas with shape
                (self.sampler.num_bboxes, 4).
            ms.Tensor: Localization target delta with shape
                (self.sampler.num_expected_neg + self.sampler.num_expected_pos, 4)
                if self.rcnn_mode else localization mask with shape
                (self.sampler.num_bboxes, ) (show what bbox was sampled for
                training).
            ms.Tensor: Classification targets. Tensor has a fixed shape
                (self.sampler.num_expected_neg + self.sampler.num_expected_pos,) if
                self.rcnn_mode else (self.sampler.num_bboxes). Show labels of
                positive bboxes.
            ms.Tensor: Classification mask. Tensor has a fixed shape
                (self.sampler.num_expected_neg + self.sampler.num_expected_pos,) if
                self.rcnn_mode else (self.sampler.num_bboxes). Show what labels
                are valid (maybe used to train classifier).
            ms.Tensor: Sampled positive bboxes with shape
                (self.sampler.num_expected_pos, 4) if self.rcnn_mode.
            ms.Tensor: Localization target delta with shape
                (self.sampler.num_expected_neg + self.sampler.num_expected_pos, 4)
                if self.rcnn_mode else localization mask with shape
                (self.sampler.num_bboxes, ) (show what bbox was sampled for
                training).
            ms.Tensor: Classification targets. Tensor has a fixed shape
                (self.sampler.num_expected_neg + self.sampler.num_expected_pos,) if
                self.rcnn_mode else (self.sampler.num_bboxes). Show labels of
                positive bboxes.
            ms.Tensor: Classification mask. Tensor has a fixed shape
                (self.sampler.num_expected_neg + self.sampler.num_expected_pos,) if
                self.rcnn_mode else (self.sampler.num_bboxes). Show what labels
                are valid (maybe used to train classifier).
        """
        gt_bboxes_i, bboxes = self.prepare(
            gt_bboxes_i=gt_bboxes_i, gt_valids=gt_valids, bboxes=bboxes,
            valid_mask=valid_mask
        )
        (
            assigned_gt_inds, overlaps, max_overlaps, assigned_labels
        ) = self.assigner(
            gt_bboxes_i=gt_bboxes_i, gt_labels_i=gt_labels_i,
            valid_mask=valid_mask, bboxes=bboxes
        )
        if self.sampler.add_gt_as_proposals:
            bboxes, assigned_gt_inds, max_overlaps = self.apply_gt_proposals(
                bboxes, gt_bboxes_i, gt_valids, assigned_gt_inds, max_overlaps
            )
        pos_index, pos_valid_index, neg_index, neg_valid_index = self.sampler(
            assigned_gt_inds=assigned_gt_inds, overlaps=overlaps,
            max_overlaps=max_overlaps, assigned_labels=assigned_labels,
            gt_bboxes_i=gt_bboxes_i, gt_labels_i=gt_labels_i,
            valid_mask=valid_mask, bboxes=bboxes, gt_valids=gt_valids
        )

        return self.get_result(
            assigned_gt_inds=assigned_gt_inds, bboxes=bboxes,
            gt_labels_i=gt_labels_i, gt_bboxes_i=gt_bboxes_i,
            pos_index=pos_index, pos_valid_index=pos_valid_index,
            neg_index=neg_index, neg_valid_index=neg_valid_index
        )
