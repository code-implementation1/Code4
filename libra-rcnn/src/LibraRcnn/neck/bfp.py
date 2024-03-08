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
"""Balanced Feature Pyramid."""
from typing import List

import mindspore as ms
from mindspore import nn

from .non_local_2d import NonLocal2d
from .conv_module import ConvModule


class BFP(nn.Cell):
    """BFP (Balanced Feature Pyramids)
    BFP takes multi-level features as inputs and gather them into a single one,
    then refine the gathered feature and scatter the refined results to
    multi-level features. This module is used in Libra R-CNN (CVPR 2019), see
    the paper `Libra R-CNN: Towards Balanced Learning for Object Detection
    <https://arxiv.org/abs/1904.02701>`_ for details.
    Args:
        in_channels (int): Number of input channels (feature maps of all levels
            should have the same channels).
        num_levels (int): Number of input feature levels.
        refine_level (int): Index of integration and refine level of BSF in
            multi-level features from bottom to top.
        refine_type (str): Type of the refine op, currently support
            [None, 'conv', 'non_local'].
    """

    def __init__(
            self,
            feature_shapes: List,
            in_channels: int = 256,
            num_levels: int = 5,
            refine_level: int = 2,
            refine_type: str = 'non_local',
    ):
        """Init BFP."""
        super(BFP, self).__init__()
        assert refine_type in [None, 'conv', 'non_local']

        self.in_channels = in_channels
        self.num_levels = num_levels
        self.feature_shapes = [tuple(shape) for shape in feature_shapes]
        self.refine_level = refine_level
        self.refine_type = refine_type
        self.conv_cfg = dict(type='Conv2d')
        self.norm_cfg = None
        assert 0 <= self.refine_level < self.num_levels

        if self.refine_type == 'conv':
            self.refine = ConvModule(
                self.in_channels,
                self.in_channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)
        elif self.refine_type == 'non_local':
            self.refine = NonLocal2d(
                self.in_channels,
                reduction=1,
                use_scale=False,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg)

        self.interpolates = [
            ms.ops.ResizeNearestNeighbor(size=shape)
            for i, shape in enumerate(self.feature_shapes)
        ]

        self.bfs = ms.ops.zeros(
            (1, in_channels, *self.feature_shapes[self.refine_level]),
            ms.float32
        )

    def construct(self, inputs: List[ms.Tensor]) -> List[ms.Tensor]:
        """Forward function."""
        assert len(inputs) == self.num_levels
        # step 1: gather multi-level features by resize and average
        gather_size = self.feature_shapes[self.refine_level]
        interpolate = self.interpolates[self.refine_level]

        bfs = self.bfs
        for i in range(self.num_levels):
            if i < self.refine_level:
                gathered, _ = ms.ops.adaptive_max_pool2d(
                    inputs[i], output_size=gather_size, return_indices=True
                )
            else:
                gathered = interpolate(inputs[i])
            bfs = bfs + gathered

        bsf = bfs / self.num_levels

        # step 2: refine gathered features
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_levels):
            out_size = inputs[i].shape[2:]
            if i < self.refine_level:
                residual = self.interpolates[i](bsf)
            else:
                residual, _ = ms.ops.adaptive_max_pool2d(
                    bsf, output_size=out_size, return_indices=True
                )
            outs.append(residual + inputs[i])

        return outs
