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
"""Bounding boxes coder-decoder."""
from typing import Optional, Sequence

import mindspore as ms
from mindspore import nn
from mindspore import ops


class DeltaXYWHBBoxCoder(nn.Cell):
    """Delta XYWH BBox coder.
    Following the practice in `R-CNN <https://arxiv.org/abs/1311.2524>`_,
    this coder encodes bbox (x1, y1, x2, y2) into delta (dx, dy, dw, dh) and
    decodes delta (dx, dy, dw, dh) back to original bbox (x1, y1, x2, y2).
    Args:
        target_means (Sequence[float]): Denormalizing means of target for
            delta coordinates
        target_stds (Sequence[float]): Denormalizing standard deviation of
            target for delta coordinates
        clip_border (bool): Whether clip the objects outside the
            border of the image. Defaults to True.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.
    """

    def __init__(
            self,
            target_means: Sequence[float] = (0., 0., 0., 0.),
            target_stds: Sequence[float] = (1., 1., 1., 1.),
            clip_border: bool = True,
            add_ctr_clamp: bool = False,
            ctr_clamp: int = 32
    ):
        super(DeltaXYWHBBoxCoder, self).__init__()
        self.means = ops.Tensor(target_means, ms.float32).reshape(1, 4)
        self.stds = ops.Tensor(target_stds, ms.float32).reshape(1, 4)
        self.clip_border = clip_border
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp
        self.eps = ms.Tensor(1e-10, ms.float32)
        self.wp_ratio_clip = ops.Tensor(0.016)

    def encode(self, bboxes: ms.Tensor, gt_bboxes: ms.Tensor) -> ms.Tensor:
        """Get box regression transformation deltas that can be used to
        transform the ``bboxes`` into the ``gt_bboxes``.
        Args:
            bboxes (ms.Tensor): Source boxes, e.g., object proposals.
            gt_bboxes (ms.Tensor): Target of the transformation, e.g.,
                ground-truth boxes.
        Returns:
            ms.Tensor: Box transformation deltas
        """
        assert bboxes.shape[0] == gt_bboxes.shape[0]
        assert bboxes.shape[-1] == 4 and gt_bboxes.shape[-1] == 4
        encoded_bboxes = self.bbox2delta(bboxes, gt_bboxes)
        return encoded_bboxes

    def decode(
            self,
            bboxes: ms.Tensor,
            pred_bboxes: ms.Tensor,
            max_shape: Optional[Sequence[int]] = None,
            wh_ratio_clip: Optional[float] = None
    ) -> ms.Tensor:
        """Apply transformation `pred_bboxes` to `boxes`.
        Args:
            bboxes (ms.Tensor): Basic boxes. Shape (N, 4) or (N, 4)
            pred_bboxes (ms.Tensor): Encoded offsets with respect to
                each roi. Has shape (N, num_classes * 4) or (N, 4) or
                (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
                when rois is a grid of anchors.Offset encoding follows [1]_.
            max_shape (Sequence[int] or mindspore.Tensor or Sequence[
                Sequence[int]],optional): Maximum bounds for boxes, specifies
                (H, W, C) or (H, W).
            wh_ratio_clip (float, optional): The allowed ratio between
                width and height.
        Returns:
            ms.Tensor: Decoded boxes.
        """
        assert pred_bboxes.shape[0] == bboxes.shape[0]
        if len(pred_bboxes.shape) == 3:
            assert pred_bboxes.shape[1] == bboxes.shape[1]
        if not wh_ratio_clip:
            wh_ratio_clip = self.wp_ratio_clip
        decoded_bboxes = self.delta2bbox(
            bboxes, pred_bboxes, max_shape,
            wh_ratio_clip, self.clip_border, self.add_ctr_clamp, self.ctr_clamp
        )

        return decoded_bboxes

    def bbox2delta(self, proposals: ms.Tensor, gt: ms.Tensor) -> ms.Tensor:
        """Compute deltas of proposals w.r.t. gt.
        We usually compute the deltas of x, y, w, h of proposals w.r.t ground
        truth bboxes to get regression target.
        This is the inverse function of :func:`delta2bbox`.

        Args:
            proposals (ms.Tensor): Boxes to be transformed, shape
                (N, ..., 4)
            gt (ms.Tensor): Gt bboxes to be used as base, shape
                (N, ..., 4)

        Returns:
            ms.Tensor: deltas with shape (N, 4), where columns represent
                dx, dy, dw, dh.
        """
        assert proposals.shape == gt.shape
        means = self.means
        stds = self.stds
        proposals = ops.cast(proposals, ms.float32)
        gt = ops.cast(gt, ms.float32)
        px = (proposals[::, 0] + proposals[::, 2]) * 0.5
        py = (proposals[::, 1] + proposals[::, 3]) * 0.5
        pw = proposals[::, 2] - proposals[::, 0] + self.eps
        ph = proposals[::, 3] - proposals[::, 1] + self.eps
        gx = (gt[::, 0] + gt[::, 2]) * 0.5
        gy = (gt[::, 1] + gt[::, 3]) * 0.5
        gw = gt[::, 2] - gt[::, 0] + self.eps
        gh = gt[::, 3] - gt[::, 1] + self.eps

        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = ops.log(gw / pw)
        dh = ops.log(gh / ph)
        deltas = ops.stack([dx, dy, dw, dh], axis=-1)

        deltas = (deltas - means) / stds

        return deltas

    def delta2bbox(
            self,
            rois: ms.Tensor,
            deltas: ms.Tensor,
            max_shape: Optional[Sequence[int]] = None,
            wh_ratio_clip: ms.Tensor = None,
            clip_border: bool = True,
            add_ctr_clamp: bool = False,
            ctr_clamp: int = 32
    ) -> ms.Tensor:
        """Apply deltas to shift/scale base boxes.
        Typically, the rois are anchor or proposed bounding boxes and the deltas
        are network outputs used to shift/scale those boxes.
        This is the inverse function of :func:`bbox2delta`.
        Args:
            rois (ms.Tensor): Boxes to be transformed. Has shape (N, 4).
            deltas (ms.Tensor): Encoded offsets relative to each roi.
                Has shape (N, num_classes * 4) or (N, 4). Note
                N = num_base_anchors * W * H, when rois is a grid of
                anchors. Offset encoding follows [1]_.
            max_shape (Optional[Sequence[int]]): Maximum bounds for boxes,
                specifies (H, W). Default None.
            wh_ratio_clip (ms.Tensor[float]): Maximum aspect ratio for boxes.
                Default 16 / 1000.
            clip_border (bool): Whether clip the objects outside the
                border of the image. Default True.
            add_ctr_clamp (bool): Whether to add center clamp. When set to True,
                the center of the prediction bounding box will be clamped to
                avoid being too far away from the center of the anchor.
                Only used by YOLOF. Default False.
            ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
                Default 32.
        Returns:
            ms.Tensor: Boxes with shape (N, num_classes * 4) or (N, 4), where 4
               represent tl_x, tl_y, br_x, br_y.
        References:
            .. [1] <https://arxiv.org/abs/1311.2524>

        Example:
            >>> rois = ms.Tensor([[ 0.,  0.,  1.,  1.],
            >>>                      [ 0.,  0.,  1.,  1.],
            >>>                      [ 0.,  0.,  1.,  1.],
            >>>                      [ 5.,  5.,  5.,  5.]])
            >>> deltas = ms.Tensor([[  0.,   0.,   0.,   0.],
            >>>                        [  1.,   1.,   1.,   1.],
            >>>                        [  0.,   0.,   2.,  -1.],
            >>>                        [ 0.7, -1.9, -0.5,  0.3]])
            >>> delta2bbox(rois, deltas, max_shape=(32, 32, 3))
            tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                    [0.1409, 0.1409, 2.8591, 2.8591],
                    [0.0000, 0.3161, 4.1945, 0.6839],
                    [5.0000, 5.0000, 5.0000, 5.0000]])
        """
        means = self.means
        stds = self.stds
        num_bboxes = deltas.shape[0]
        num_classes = deltas.shape[1] // 4
        deltas = ops.reshape(deltas, (-1, 4))

        denorm_deltas = deltas * stds + means

        dxy = denorm_deltas[:, :2]
        dwh = denorm_deltas[:, 2:]

        # Compute width/height of each roi
        rois_ = rois.repeat(1, num_classes).reshape(-1, 4)
        pxy = ((rois_[:, :2] + rois_[:, 2:]) * 0.5)
        pwh = (rois_[:, 2:] - rois_[:, :2])

        dxy_wh = pwh * dxy

        max_ratio = ops.abs(ops.log(wh_ratio_clip))
        if add_ctr_clamp:
            dxy_wh = ops.clip_by_value(
                dxy_wh, clip_value_min=ctr_clamp, clip_value_max=-ctr_clamp
            )
            dwh = ops.clip_by_value(dwh, clip_value_max=max_ratio)
        else:
            dwh = ops.clip_by_value(
                dwh, clip_value_min=-max_ratio, clip_value_max=max_ratio
            )

        gxy = pxy + dxy_wh
        gwh = pwh * ops.exp(dwh)
        x1y1 = gxy - (gwh * 0.5)
        x2y2 = gxy + (gwh * 0.5)
        bboxes = ops.concat([x1y1, x2y2], axis=-1)
        if clip_border and max_shape is not None:
            bboxes[..., 0::2] = ops.clip_by_value(
                bboxes[..., 0::2], clip_value_min=0,
                clip_value_max=max_shape[1]
            )
            bboxes[..., 1::2] = ops.clip_by_value(
                bboxes[..., 1::2], clip_value_min=0,
                clip_value_max=max_shape[0]
            )
        bboxes = ops.reshape(bboxes, (num_bboxes, 4))
        return bboxes
