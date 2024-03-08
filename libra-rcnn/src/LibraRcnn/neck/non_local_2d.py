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
# This file has been derived from the https://github.com/open-mmlab/mmcv/tree/v1.7.1
# repository and modified.
# ============================================================================
"""Non Local Module."""
from typing import Dict, Optional

import mindspore as ms
from mindspore import nn

from .conv_module import ConvModule


class _NonLocalNd(nn.Cell):
    """Basic Non-local module.

    This module is proposed in
    "Non-local Neural Networks"
    Paper reference: https://arxiv.org/abs/1711.07971

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio. Default: 2.
        use_scale (bool): Whether to scale pairwise_weight by
            `1/sqrt(inter_channels)` when the mode is `embedded_gaussian`.
            Default: True.
        conv_cfg (None | dict): The config dict for convolution layers.
            If not specified, it will use `nn.Conv2d` for convolution layers.
            Default: None.
        norm_cfg (None | dict): The config dict for normalization layers.
            Default: None. (This parameter is only applicable to conv_out.)
        mode (str): Options are `gaussian`, `concatenation`,
            `embedded_gaussian` and `dot_product`. Default: embedded_gaussian.
    """

    def __init__(self,
                 in_channels: int,
                 reduction: int = 2,
                 use_scale: bool = True,
                 conv_cfg: Optional[Dict] = None,
                 norm_cfg: Optional[Dict] = None,
                 mode: str = 'embedded_gaussian',
                 **kwargs):
        """Init _NonLocalNd."""
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = max(in_channels // reduction, 1)
        self.mode = mode

        if mode not in [
                'gaussian', 'embedded_gaussian', 'dot_product', 'concatenation'
        ]:
            raise ValueError("Mode should be in 'gaussian', 'concatenation', "
                             f"'embedded_gaussian' or 'dot_product', but got "
                             f'{mode} instead.')

        # g, theta, phi are defaulted as `nn.ConvNd`.
        # Here we use ConvModule for potential usage.
        self.g = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            act_cfg=None)  # type: ignore
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        if self.mode != 'gaussian':
            self.theta = ConvModule(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                act_cfg=None)
            self.phi = ConvModule(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                act_cfg=None)

        if self.mode == 'concatenation':
            self.concat_project = ConvModule(
                self.inter_channels * 2,
                1,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
                act_cfg=dict(type='ReLU'))

        self.concat_1 = ms.ops.Concat(axis=1)
        self.reshape = ms.ops.Reshape()

        self.pairwise_func = getattr(self, self.mode)

    def gaussian(self, theta_x: ms.Tensor,
                 phi_x: ms.Tensor) -> ms.Tensor:
        """Call when gaussian mode."""
        # NonLocal1d pairwise_weight: [N, H, H]
        pairwise_weight = ms.ops.matmul(theta_x, phi_x)
        pairwise_weight = ms.ops.softmax(pairwise_weight, axis=-1)
        return pairwise_weight

    def embedded_gaussian(self, theta_x: ms.Tensor,
                          phi_x: ms.Tensor) -> ms.Tensor:
        """Call when embedded_gaussian mode."""
        # NonLocal1d pairwise_weight: [N, H, H]
        pairwise_weight = ms.ops.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            pairwise_weight /= theta_x.shape[-1] ** 0.5
        pairwise_weight = ms.ops.softmax(pairwise_weight, axis=-1)
        return pairwise_weight

    def dot_product(self, theta_x: ms.Tensor,
                    phi_x: ms.Tensor) -> ms.Tensor:
        """Call when dot_product mode."""
        # NonLocal1d pairwise_weight: [N, H, H]
        pairwise_weight = ms.ops.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def concatenation(self, theta_x: ms.Tensor,
                      phi_x: ms.Tensor) -> ms.Tensor:
        """Call when concatenation mode."""
        # NonLocal1d pairwise_weight: [N, H, H]
        h = theta_x.shape[2]
        w = phi_x.shape[3]
        theta_x = ms.ops.repeat_elements(theta_x, rep=w, axis=3)
        phi_x = ms.ops.repeat_elements(phi_x, rep=h, axis=2)

        concat_feature = self.concat_1([theta_x, phi_x])
        pairwise_weight = self.concat_project(concat_feature)
        n, _, h, w = pairwise_weight.shape
        pairwise_weight = pairwise_weight.reshape(n, h, w)
        pairwise_weight /= pairwise_weight.shape[-1]

        return pairwise_weight

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """Forward function."""
        # Assume `reduction = 1`, then `inter_channels = C`
        # or `inter_channels = C` when `mode="gaussian"`

        # NonLocal1d x: [N, C, H]
        n = x.shape[0]

        # NonLocal1d g_x: [N, H, C]
        g_x = self.reshape(self.g(x), (n, self.inter_channels, -1))
        g_x = ms.ops.transpose(g_x, (0, 2, 1))

        # NonLocal1d theta_x: [N, H, C], phi_x: [N, C, H]
        if self.mode == 'gaussian':
            theta_x = self.reshape(x, (n, self.in_channels, -1))
            theta_x = ms.ops.transpose(theta_x, (0, 2, 1))
            if self.sub_sample:
                phi_x = self.reshape(self.phi(x), (n, self.in_channels, -1))
            else:
                phi_x = self.reshape(x, (n, self.in_channels, -1))
        elif self.mode == 'concatenation':
            theta_x = self.reshape(self.theta(x),
                                   (n, self.inter_channels, -1, 1))
            phi_x = self.reshape(self.phi(x), (n, self.inter_channels, 1, -1))
        else:
            theta_x = self.reshape(self.theta(x), (n, self.inter_channels, -1))
            theta_x = ms.ops.transpose(theta_x, (0, 2, 1))
            phi_x = self.reshape(self.phi(x), (n, self.inter_channels, -1))

        # NonLocal1d pairwise_weight: [N, H, H]
        pairwise_weight = self.pairwise_func(theta_x, phi_x)

        # NonLocal1d y: [N, H, C]
        y = ms.ops.matmul(pairwise_weight, g_x)
        # NonLocal1d y: [N, C, H]
        y = self.reshape(
            ms.ops.transpose(y, (0, 2, 1)),
            (n, self.inter_channels) + x.shape[2:]
        )
        output = x + self.conv_out(y)

        return output


class NonLocal2d(_NonLocalNd):
    """2D Non-local module.

    Args:
        in_channels (int): Same as `NonLocalND`.
        sub_sample (bool): Whether to apply max pooling after pairwise
            function (Note that the `sub_sample` is applied on spatial only).
            Default: False.
        conv_cfg (None | dict): Same as `NonLocalND`.
            Default: dict(type='conv2d').
    """

    _abbr_ = 'nonlocal_block'

    def __init__(
            self,
            in_channels: int,
            sub_sample: bool = False,
            conv_cfg: Optional[Dict] = None,
            **kwargs
    ):
        """Init NonLocal2d."""
        if conv_cfg is None:
            conv_cfg = dict(type='conv2d')
        super().__init__(in_channels, conv_cfg=conv_cfg, **kwargs)

        self.sub_sample = sub_sample

        if sub_sample:
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            self.g = nn.SequentialCell(self.g, max_pool_layer)
            if self.mode != 'gaussian':
                self.phi = nn.SequentialCell(self.phi, max_pool_layer)
            else:
                self.phi = max_pool_layer
