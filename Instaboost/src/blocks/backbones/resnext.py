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
# and modified: https://github.com/open-mmlab/mmdetection
# ============================================================================
"""ResNext backbone"""
from typing import ClassVar
import math

import mindspore as ms
from mindspore import nn

from .resnet import ResNet


class Bottleneck(nn.Cell):
    """Bottleneck ResNext block

    Args:
        inplanes (int): number of input channels
        planes (int): number of output channels
        stride (int): stride size in convolution
        dilation (int): dilation size in convolution
        downsample (ms.nn.SequentialCell): conv+bn block for residual
        norm_eval (bool): if True, BN layer will have only evaluation behaviour
        weights_update (bool): if False, all convolution layer will be frozen.
        groups (int): number groups in convolution.
        base_width (int): base width of resnext.
    """

    expansion = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            dilation: int = 1,
            downsample: nn.Cell = None,
            norm_eval: bool = False,
            weights_update: bool = False,
            groups: int = 1,
            base_width: int = 4
    ):
        """Init bottleneck block for ResNeXt."""
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.conv1_stride = 1
        self.conv2_stride = stride

        self.weights_update = weights_update
        self.norm_eval = norm_eval
        self.affine = weights_update

        if groups == 1:
            width = self.planes
        else:
            width = math.floor(self.planes * (base_width / 64)) * groups

        self.bn1 = nn.BatchNorm2d(
            width, affine=self.affine,
            use_batch_statistics=False if self.norm_eval else None
        )
        self.bn2 = nn.BatchNorm2d(
            width, affine=self.affine,
            use_batch_statistics=False if self.norm_eval else None
        )
        self.bn3 = nn.BatchNorm2d(
            self.planes * self.expansion, affine=self.affine,
            use_batch_statistics=False if self.norm_eval else None
        )

        self.conv1 = nn.Conv2d(
            self.inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            has_bias=False
        )
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=self.dilation,
            pad_mode='pad',
            dilation=self.dilation,
            has_bias=False,
            group=groups,
        )
        self.conv3 = nn.Conv2d(
            width,
            self.planes * self.expansion,
            kernel_size=1,
            has_bias=False
        )

        self.relu = nn.ReLU()
        self.downsample = downsample

        if not self.weights_update:
            self.conv1.weight.requires_grad = False
            self.conv2.weight.requires_grad = False
            self.conv3.weight.requires_grad = False
            if self.downsample is not None:
                self.downsample[0].weight.requires_grad = False

        if self.norm_eval and self.downsample is not None:
            self.downsample[1].use_batch_statistics = False

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """Run forward the bottleneck block.

        Args:
            x (ms.Tensor): Feature map of shape (b, c, h, w).

        Returns:
            ms.Tensor:
                Output feature map.
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class ResNeXt(ResNet):
    """ResNeXt backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        groups (int): Group of resnext.
        base_width (int): Base width of resnext.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self, groups: int = 1, base_width: int = 4, **kwargs):
        """Init resnext."""
        super(ResNeXt, self).__init__(**kwargs)
        self.groups = groups
        self.base_width = base_width

        self.inplanes = 64
        self.res_layers = nn.CellList()
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            planes = 64 * 2 ** i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                base_width=self.base_width,
                groups=self.groups)
            self.inplanes = planes * self.block.expansion
            self.res_layers.append(res_layer)

        self.feat_dim = self.block.expansion * 64 * 2 ** (
            len(self.stage_blocks) - 1
        )


def make_res_layer(
        block: ClassVar,
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        base_width: int = 4,
        weights_update: bool = True,
        norm_eval: bool = False
):
    """Make residual block."""
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.SequentialCell(
            nn.Conv2d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                has_bias=False),
            nn.BatchNorm2d(planes * block.expansion, affine=weights_update,
                           use_batch_statistics=False if norm_eval else None),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            stride,
            dilation,
            downsample,
            norm_eval=norm_eval,
            weights_update=weights_update,
            groups=groups,
            base_width=base_width
        )
    )
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                dilation=dilation,
                stride=1,
                downsample=None,
                norm_eval=norm_eval,
                weights_update=weights_update,
                groups=groups,
                base_width=base_width
            )
        )

    return nn.SequentialCell(*layers)
