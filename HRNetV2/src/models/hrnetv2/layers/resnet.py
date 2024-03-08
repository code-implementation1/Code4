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
"""Resnet backbone."""
from typing import Sequence, Tuple, ClassVar

import mindspore as ms
from mindspore import nn


class BasicBlock(nn.Cell):
    """Basic ResNet block

    Args:
        inplanes (int): number of input channels.
        planes (int): number of output channels.
        stride (int): stride size in convolution.
        dilation (int): dilation size in convolution.
        downsample (ms.nn.SequentialCell): conv+bn block for residual.
        norm_eval (bool): if True, BN layer will have only evaluation behaviour.
        weights_update (bool): if False, all convolution layer will be frozen.
    """

    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            dilation: int = 1,
            downsample: nn.SequentialCell = None,
            norm_eval: bool = False,
            weights_update: bool = False,
    ):
        super(BasicBlock, self).__init__()

        self.weights_update = weights_update
        self.norm_eval = norm_eval
        self.affine = weights_update

        self.bn1 = nn.BatchNorm2d(
            planes, affine=self.affine,
            use_batch_statistics=False if self.norm_eval else None
        )
        self.bn2 = nn.BatchNorm2d(
            planes, affine=self.affine,
            use_batch_statistics=False if self.norm_eval else None
        )

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            pad_mode='pad',
            dilation=dilation,
            has_bias=False
        )

        self.conv2 = nn.Conv2d(
            planes, planes, 3, padding=1, pad_mode='pad',
            has_bias=False
        )

        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

        if not self.weights_update:
            self.conv1.weight.requires_grad = False
            self.conv2.weight.requires_grad = False
            if self.downsample is not None:
                self.downsample[0].weight.requires_grad = False

        if self.norm_eval and self.downsample is not None:
            self.downsample[1].use_batch_statistics = False

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """Run forward the basic block.

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

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    """Bottleneck block for ResNet.

    Args:
        inplanes (int): number of input channels
        planes (int): number of output channels
        stride (int): stride size in convolution
        dilation (int): dilation size in convolution
        downsample (ms.nn.SequentialCell): conv+bn block for residual
        norm_eval (bool): if True, BN layer will have only evaluation behaviour
        weights_update (bool): if False, all convolution layer will be frozen.
    """

    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            dilation=1,
            downsample=None,
            norm_eval=False,
            weights_update=False,
    ):
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

        self.bn1 = nn.BatchNorm2d(
            planes, affine=self.affine,
            use_batch_statistics=False if self.norm_eval else None
        )
        self.bn2 = nn.BatchNorm2d(
            planes, affine=self.affine,
            use_batch_statistics=False if self.norm_eval else None
        )
        self.bn3 = nn.BatchNorm2d(
            planes * self.expansion, affine=self.affine,
            use_batch_statistics=False if self.norm_eval else None
        )

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            has_bias=False
        )
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            pad_mode='pad',
            dilation=dilation,
            has_bias=False
        )
        self.conv3 = nn.Conv2d(
            planes,
            planes * self.expansion,
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


class ResNet(nn.Cell):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            "-1" means not freezing any parameters.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(
            self,
            depth: int,
            num_stages: int = 4,
            strides: Sequence[int] = (1, 2, 2, 2),
            dilations: Sequence[int] = (1, 1, 1, 1),
            out_indices: Sequence[int] = (0, 1, 2, 3),
            frozen_stages: int = -1,
            norm_eval: bool = False
    ):
        """Init resnet."""
        super(ResNet, self).__init__()
        self.depth = depth
        self.num_stages = num_stages
        assert 4 >= num_stages >= 1
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self._make_stem_layer()

        self.res_layers = nn.CellList()
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = 64 * 2 ** i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                weights_update=(i > self.frozen_stages),
                norm_eval=norm_eval
            )
            self.inplanes = planes * self.block.expansion
            self.res_layers.append(res_layer)

        self.feat_dim = self.block.expansion * 64 * 2 ** (
            len(self.stage_blocks) - 1
        )

    def _make_stem_layer(self):
        """Make stem layer"""
        weights_update = self.frozen_stages >= 0
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            pad_mode='pad',
            has_bias=False
        )
        self.bn1 = nn.BatchNorm2d(
            64, affine=weights_update,
            use_batch_statistics=False if self.norm_eval else None
        )

        if weights_update:
            self.conv1.weight.requires_grad = False

        self.relu = nn.ReLU()

        self.maxpool = nn.SequentialCell(
            nn.Pad(((0, 0), (0, 0), (1, 1), (1, 1))),
            nn.MaxPool2d(
                kernel_size=3, stride=2, pad_mode='valid'
            )
        )

    def construct(self, x: ms.Tensor) -> Tuple:
        """Forward resnet.

        Args:
            x: Images of shape (n, 3, h, w)

        Returns:
            Tuple:
                List of output feature maps (ms.Tensor), extracted from images.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, res_layer in enumerate(self.res_layers):
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def set_train(self, mode: bool = True):
        super(ResNet, self).set_train(mode=mode)
        for res_layer in self.res_layers:
            for block in res_layer:
                block.set_train(mode)


def make_res_layer(
        block: ClassVar,
        inplanes: int,
        planes: int,
        blocks: int,
        stride: int = 1,
        dilation: int = 1,
        weights_update: bool = True,
        norm_eval: bool = False
) -> nn.Cell:
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

    layers = [block(
        inplanes,
        planes,
        stride,
        dilation,
        downsample,
        norm_eval=norm_eval,
        weights_update=weights_update
    )]
    inplanes = planes * block.expansion
    for _ in range(1, blocks):
        layers.append(
            block(
                inplanes,
                planes,
                1,
                dilation,
                norm_eval=norm_eval,
                weights_update=weights_update
            )
        )

    return nn.SequentialCell(*layers)
