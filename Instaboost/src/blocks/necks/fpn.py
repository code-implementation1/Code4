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
"""Feature pyramid network."""
from typing import List, Sequence, Tuple, Union

import mindspore as ms
from mindspore import nn
from mindspore import ops

from ..layers import ConvModule
from .. import Config


class FPN(nn.Cell):
    r"""Feature Pyramid Network.
    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.
    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Defaults to 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Defaults to -1, which means the
            last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Defaults to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed
            * 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            * 'on_lateral': Last feature map after lateral convs.
            * 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Defaults to False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Defaults to False.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Defaults to None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Defaults to None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Defaults to None.

    Examples:
        >>> in_channels = [256, 512, 1024, 2048]
        >>> out_channels = 2
        >>> num_outs = 5
        >>> feature_shapes = [[256 // 2 ** i] * 2 for i in range(len(in_channels))]
        >>> tensors = [
        ...     ms.Tensor(np.random.randn(1, ch, *fs), dtype=ms.float32)
        ...     for fs, ch in zip(feature_shapes, in_channels)
        ... ]
        >>> fpn = FPN(
        ...    feature_shapes=feature_shapes, in_channels=in_channels,
        ...    out_channels=out_channels, num_outs=num_outs
        ... )
        >>> outputs = fpn(tensors)
        >>> len(outputs)
        5
        >>> [a.shape for a in outputs]
        [(1, 2, 256, 256), (1, 2, 128, 128), (1, 2, 64, 64), (1, 2, 32, 32), (1, 2, 16, 16)]
        >>> fpn = FPN(
        ...     feature_shapes=feature_shapes, in_channels=in_channels,
        ...     out_channels=out_channels, num_outs=4, start_level=1
        ... )
        >>> outputs = fpn(tensors)
        >>> len(outputs)
        4
        >>> [a.shape for a in outputs]
        [(1, 2, 128, 128), (1, 2, 64, 64), (1, 2, 32, 32), (1, 2, 16, 16)]
    """

    def __init__(
            self,
            feature_shapes: List,
            in_channels: List[int],
            out_channels: int,
            num_outs: int,
            start_level: int = 0,
            end_level: int = -1,
            add_extra_convs: Union[bool, str] = False,
            relu_before_extra_convs: bool = False,
            no_norm_on_lateral: bool = False,
            conv_cfg: Config = None,
            norm_cfg: Config = None,
            act_cfg: Config = None
    ) -> None:
        """Init FPN"""
        super().__init__()
        assert isinstance(in_channels, list)
        conv_cfg = conv_cfg.as_dict() if conv_cfg else None
        norm_cfg = norm_cfg.as_dict() if norm_cfg else None
        act_cfg = act_cfg.as_dict() if act_cfg else None

        self.feature_shapes = feature_shapes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral

        if end_level in (-1, end_level == self.num_ins - 1):
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        lateral_convs = []
        fpn_convs = []

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            lateral_convs.append(l_conv)
            fpn_convs.append(fpn_conv)
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                fpn_convs.append(extra_fpn_conv)

        self.lateral_convs = nn.CellList(lateral_convs)
        self.fpn_convs = nn.CellList(fpn_convs)

        self.interpolations = []
        for shape in self.feature_shapes[self.start_level:self.backbone_end_level]:
            self.interpolations.append(ops.ResizeNearestNeighbor(shape))
        self.maxpool = ops.MaxPool(kernel_size=1, strides=2)

    def construct(self, inputs: Sequence[ms.Tensor]) -> Tuple:
        """Forward function.

        Args:
            inputs: List of input feature maps. Shapes
            (n, self.in_channels[i], self.feature_shapes[i][0], self.feature_shapes[i][1]).

        Returns:
            Tuple: Tuple of output tensors.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals_ = [
            self.lateral_convs[i](inputs[i + self.start_level])
            for i in range(len(self.lateral_convs))
        ]

        # build top-down path
        used_backbone_levels = len(laterals_)
        laterals = []
        for i in range(used_backbone_levels - 1, -1, -1):
            if i == used_backbone_levels - 1:
                laterals.append(laterals_[i])
            else:
                laterals = [
                    laterals_[i] + self.interpolations[i](laterals[0]),
                ] + laterals
        # build outputs
        # part 1: from original levels
        outs = []
        for i in range(used_backbone_levels):
            outs.append(self.fpn_convs[i](laterals[i]))
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(self.maxpool(outs[-1]))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](ops.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)
