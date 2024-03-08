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
"""Anchor generator."""
from typing import List, Optional, Tuple, Union

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import ops


def _pair(x: Union[Tuple[int, int], int]):
    return (x, x) if isinstance(x, int) else x


class AnchorGenerator(nn.Cell):
    """Standard anchor generator for 2D anchor-based detectors.
    Args:
        strides (Union[List[int], List[Tuple[int, int]]]): Strides of anchors
            in multiple feature levels in order (w, h).
        ratios (List[float]): The list of ratios between the height and width
            of anchors in a single level.
        scales (Optional[List[Union[int, float]]]): Anchor scales for anchors
            in a single level. It cannot be set at the same time if
            `octave_base_scale` and `scales_per_octave` are set.
        base_sizes (Optional[List[int]]): The basic sizes
            of anchors in multiple levels.
            If None is given, strides will be used as base_sizes.
            (If strides are non-square, the shortest stride is taken.)
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. By default, it is True in V2.0
        octave_base_scale (int): The base scale of octave.
        scales_per_octave (int): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
        centers (Optional[List[Tuple[float, float]]]): The centers of the
            anchor relative to the feature grid center in multiple feature
            levels. By default, it is set to be None and not used. If a list of
            tuple of float is given, they will be used to shift the centers of
            anchors.
        center_offset (float): The offset of center in proportion to anchors'
            width and height. By default, it is 0 in V2.0.
    Examples:
        >>> self = AnchorGenerator([16], [1.], [1.], [9])
        >>> all_anchors = self.grid_priors([(2, 2)])
        >>> print(all_anchors)
        (Tensor(shape=[4, 4], dtype=Float32, value=
        [[-4.50000000e+00, -4.50000000e+00,  4.50000000e+00,  4.50000000e+00],
         [ 1.15000000e+01, -4.50000000e+00,  2.05000000e+01,  4.50000000e+00],
         [-4.50000000e+00,  1.15000000e+01,  4.50000000e+00,  2.05000000e+01],
         [ 1.15000000e+01,  1.15000000e+01,  2.05000000e+01,  2.05000000e+01]]),)
        >>> self = AnchorGenerator([16, 32], [1.], [1.], [9, 18])
        >>> all_anchors = self.grid_priors([(2, 2), (1, 1)])
        >>> print(all_anchors)
        (Tensor(shape=[4, 4], dtype=Float32, value=
        [[-4.50000000e+00, -4.50000000e+00,  4.50000000e+00,  4.50000000e+00],
         [ 1.15000000e+01, -4.50000000e+00,  2.05000000e+01,  4.50000000e+00],
         [-4.50000000e+00,  1.15000000e+01,  4.50000000e+00,  2.05000000e+01],
         [ 1.15000000e+01,  1.15000000e+01,  2.05000000e+01,  2.05000000e+01]]),
         Tensor(shape=[1, 4], dtype=Float32, value=
        [[-9.00000000e+00, -9.00000000e+00,  9.00000000e+00,  9.00000000e+00]]))
    """

    def __init__(
            self,
            strides: Union[List[int], List[Tuple[int, int]]],
            ratios: List[float],
            scales: Optional[List[Union[int, float]]] = None,
            base_sizes: Optional[List[int]] = None,
            scale_major: bool = True,
            octave_base_scale: int = None,
            scales_per_octave: int = None,
            centers: Optional[List[Tuple[float, float]]] = None,
            center_offset: float = 0.
    ):
        # check center and center_offset
        super().__init__()
        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset' \
                                    f'!=0, {centers} is given.'
        if not 0 <= center_offset <= 1:
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(strides), \
                'The number of strides should be the same as centers, got ' \
                f'{strides} and {centers}'

        # calculate base sizes of anchors
        self.strides = [_pair(stride) for stride in strides]
        self.base_sizes = [min(stride) for stride in self.strides
                           ] if base_sizes is None else base_sizes
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'

        # calculate scales of anchors
        assert ((octave_base_scale is not None
                 and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
        if scales is not None:
            self.scales = ms.Tensor(scales, ms.float32)
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = ms.Tensor(
                np.array(
                    [
                        2 ** (i / scales_per_octave)
                        for i in range(scales_per_octave)
                    ]
                )
            )
            scales = octave_scales * octave_base_scale
            self.scales = ops.cast(scales, ms.float32)
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = ms.Tensor(ratios, ms.float32)
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

        self.one = ms.Tensor(1., ms.float32)
        self.zero = ms.Tensor(0., ms.float32)

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return self.num_base_priors

    @property
    def num_base_priors(self):
        """list[int]: The number of priors (anchors) at a point
        on the feature grid"""
        return [base_anchors.shape[0] for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    def gen_base_anchors(self):
        """Generate base anchors.
        Returns:
            list(mindspore.Tensor): Base anchors of a feature grid in multiple
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(
            self,
            base_size: Union[int, float],
            scales: ms.Tensor,
            ratios: ms.Tensor,
            center: Optional[Tuple[float]] = None
    ) -> ms.Tensor:
        """Generate base anchors of a single level.
        Args:
            base_size (Union[int, float]): Basic size of an anchor.
            scales (ms.Tensor): Scales of the anchor.
            ratios (ms.Tensor): The ratio between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.
        Returns:
            ms.Tensor: Anchors in a single-level feature maps.
        """
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = ops.sqrt(ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (
                w * ops.expand_dims(w_ratios, -1) * ops.expand_dims(scales, 0)
            ).reshape(-1)
            hs = (
                h * ops.expand_dims(h_ratios, -1) * ops.expand_dims(scales, 0)
            ).reshape(-1)
        else:
            ws = (w * ops.expand_dims(scales, -1) * ops.expand_dims(w_ratios, 0)).reshape(-1)
            hs = (h * ops.expand_dims(scales, -1) * ops.expand_dims(h_ratios, 0)).reshape(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]
        base_anchors = ops.stack(base_anchors, axis=-1)

        return base_anchors

    @ms.jit()
    def grid_priors(
            self, featmap_sizes: List[Tuple[int, int]]
    ) -> List[ms.Tensor]:
        """Generate grid anchors in multiple feature levels.
        Args:
            featmap_sizes (List[Tuple[int, int]]): List of feature map sizes in
                multiple feature levels.
        Return:
            List[ms.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature level, \
                num_base_anchors is the number of anchors for that level.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_priors(
                featmap_sizes[i], level_idx=i)
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_priors(
            self, featmap_size: Tuple[int, int], level_idx: int
    ):
        """Generate grid anchors of a single level.
        Note:
            This function is usually called by method ``self.grid_priors``.
        Args:
            featmap_size (Tuple[int, int]): Size of the feature maps.
            level_idx (int): The index of corresponding feature map level.
        Returns:
            ms.Tensor: Anchors in the overall feature maps.
        """
        base_anchors = self.base_anchors[level_idx]
        feat_h, feat_w = featmap_size
        feat_h_f = ms.Tensor(feat_h, ms.float32)
        feat_w_f = ms.Tensor(feat_w, ms.float32)
        stride_w, stride_h = self.strides[level_idx]
        # First create Range with the default dtype, than convert to
        # target `dtype` for onnx exporting.
        shift_x = ops.range(self.zero, feat_w_f, self.one) * stride_w
        shift_y = ops.range(self.zero, feat_h_f, self.one) * stride_h

        shift_xx, shift_yy = ops.meshgrid(shift_x, shift_y)
        shift_xx = ops.reshape(shift_xx, (feat_w * feat_h,))
        shift_yy = ops.reshape(shift_yy, (feat_w * feat_h,))
        shifts = ops.stack([shift_xx, shift_yy, shift_xx, shift_yy], axis=-1)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = (
            ops.expand_dims(base_anchors, 0) + ops.expand_dims(shifts, 1)
        )
        all_anchors = ops.reshape(all_anchors, (-1, 4))
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(
            self, featmap_sizes: List[Tuple[int, int]],
            pad_shape: Tuple[int, int]
    ):
        """Generate valid flags of anchors in multiple feature levels.
        Args:
            featmap_sizes (List[Tuple[int, int]): List of feature map sizes in
                multiple feature levels.
            pad_shape (Tuple[int, int]): The padded shape of the image.
        Return:
            list(mindspore.Tensor): Valid flags of anchors in multiple levels.
        """
        assert self.num_levels == len(featmap_sizes)
        multi_level_flags = []
        for i in range(self.num_levels):
            anchor_stride = self.strides[i]
            feat_h, feat_w = featmap_sizes[i]
            h, w = pad_shape[:2]
            valid_feat_h = min(int(np.ceil(h / anchor_stride[1])), feat_h)
            valid_feat_w = min(int(np.ceil(w / anchor_stride[0])), feat_w)
            flags = self.single_level_valid_flags(
                (feat_h, feat_w), (valid_feat_h, valid_feat_w),
                self.num_base_anchors[i]
            )
            multi_level_flags.append(flags)
        return multi_level_flags

    def single_level_valid_flags(
            self,
            featmap_size: Tuple[int, int],
            valid_size: Tuple[int, int],
            num_base_anchors: int,
    ):
        """Generate the valid flags of anchor in a single feature map.
        Args:
            featmap_size (Tuple[int, int]): The size of feature maps, arrange
                as (h, w).
            valid_size (Tuple[int, int]): The valid size of the feature maps.
            num_base_anchors (int): The number of base anchors.
        Returns:
            ms.Tensor: The valid flags of each anchor in a single level
            feature map.
        """
        feat_h, feat_w = featmap_size
        feat_h_f = ops.Tensor(feat_h, ms.float32)
        feat_w_f = ops.Tensor(feat_w, ms.float32)
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = ops.less(ops.range(self.zero, feat_w_f, self.one), valid_w)
        valid_y = ops.less(ops.range(self.zero, feat_h_f, self.one), valid_h)

        valid_xx, valid_yy = ops.meshgrid((valid_x, valid_y), indexing='xy')
        valid_xx = ops.reshape(valid_xx, (feat_w * feat_h,))
        valid_yy = ops.reshape(valid_yy, (feat_w * feat_h,))

        valid = ops.logical_and(valid_xx, valid_yy)
        valid = ops.reshape(
            ops.broadcast_to(
                ops.expand_dims(valid, -1),
                (valid.shape[0], num_base_anchors)
            ),
            (-1,)
        )
        return valid
