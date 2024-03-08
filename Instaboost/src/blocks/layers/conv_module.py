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
# and modified: https://github.com/open-mmlab/mmcv/tree/v1.7.1
# ============================================================================
"""Convolution module."""
import warnings
from typing import Dict, Optional, Tuple, Union

import mindspore as ms
from mindspore import nn

from ..initialization import kaiming_init, constant_init

convs = {
    'Conv2d': nn.Conv2d
}
norms = {
    'BN2d': nn.BatchNorm2d,
    'IN2d': nn.InstanceNorm2d,
    'GN': nn.GroupNorm
}
activations = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'GELU': nn.GELU,
    'LeakyReLU': nn.LeakyReLU
}
upsample_cfg = {
    'nearest': ms.ops.ResizeNearestNeighbor,
    'bilinear': ms.ops.ResizeBilinear,
    'deconv': nn.Conv2dTranspose
}


def build_conv_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Cell:
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Cell: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in convs:
        raise KeyError(f'Unrecognized layer type {layer_type}')

    conv_layer = convs.get(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


def build_activation_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Cell:
    """Build activation layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = None
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in activations:
        raise KeyError(f'Unrecognized layer type {layer_type}')

    act_layer = activations.get(layer_type)

    layer = act_layer(*args, **kwargs, **cfg_)

    return layer


def build_norm_layer(cfg: Dict, num_features: int,) -> nn.Cell:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in norms:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    norm_layer = norms.get(layer_type)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if layer_type != 'GN':
        layer = norm_layer(num_features, **cfg_)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for _, param in layer.parameters_dict(True).items():
        param.requires_grad = requires_grad

    return layer


def build_upsample_layer(cfg):
    """ Build upsample layer
    Args:
        cfg (dict): cfg should contain:
            type (str): Identify upsample layer type.
            upsample ratio (int): Upsample ratio
            layer args: args needed to instantiate a upsample layer.
    Returns:
        layer (nn.Module): Created upsample layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in upsample_cfg:
        raise KeyError('Unrecognized upsample type {}'.format(layer_type))
    upsample = upsample_cfg[layer_type]
    if upsample is None:
        raise NotImplementedError

    layer = upsample(**cfg_)
    return layer


class ConvModule(nn.Cell):
    """A conv block that bundles conv/norm/activation layers.

    This block simplifies the usage of convolution layers, which are commonly
    used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    It is based upon three build methods: `build_conv_layer()`,
    `build_norm_layer()` and `build_activation_layer()`.

    Args:
        in_channels (int): Number of channels in the input feature map.
            Same as that in ``nn._ConvNd``.
        out_channels (int): Number of channels produced by the convolution.
            Same as that in ``nn._ConvNd``.
        kernel_size (int | tuple[int]): Size of the convolving kernel.
            Same as that in ``nn._ConvNd``.
        stride (int | tuple[int]): Stride of the convolution.
            Same as that in ``nn._ConvNd``.
        padding (int | tuple[int]): Zero-padding added to both sides of
            the input. Same as that in ``nn._ConvNd``.
        dilation (int | tuple[int]): Spacing between kernel elements.
            Same as that in ``nn._ConvNd``.
        groups (int): Number of blocked connections from input channels to
            output channels. Same as that in ``nn._ConvNd``.
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None,
            which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU').
        inplace (bool): Whether to use inplace mode for activation.
            Default: True.
        padding_mode (str): Padding mode ['pad', 'valid', 'same']
            Default: 'pad'.
        order (tuple[str]): The order of conv/norm/activation layers. It is a
            sequence of "conv", "norm" and "act". Common examples are
            ("conv", "norm", "act") and ("act", "conv", "norm").
            Default: ('conv', 'norm', 'act').
    """

    _abbr_ = 'conv_block'

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: Union[bool, str] = 'auto',
            conv_cfg: Optional[Dict] = None,
            norm_cfg: Optional[Dict] = None,
            act_cfg: Optional[Dict] = 'ReLU',
            inplace: bool = True,
            padding_mode: str = 'pad',
            order: tuple = ('conv', 'norm', 'act')
    ):
        """Init ConvModule."""
        super().__init__()
        if act_cfg is not None and isinstance(act_cfg, str):
            act_cfg = dict(type=act_cfg)
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {'conv', 'norm', 'act'}
        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        # reset padding to 0 for conv module
        conv_padding = padding
        # build convolution layer
        self.conv = build_conv_layer(
            self.conv_cfg,
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            group=groups,
            has_bias=bias,
            pad_mode=padding_mode,
        )
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.pad_mode = self.conv.pad_mode
        self.group = self.conv.group

        # build normalization layers
        if self.with_norm:
            # norm layer is after conv layer
            if order.index('norm') > order.index('conv'):
                norm_channels = out_channels
            else:
                norm_channels = in_channels
            self.norm = build_norm_layer(self.norm_cfg, norm_channels)
            if self.with_bias:
                if isinstance(self.norm, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                    warnings.warn(
                        'Unnecessary conv bias before batch/instance norm')

        # build activation layer
        if self.with_activation:
            self.activate = build_activation_layer(act_cfg)

        self.ops_list = []
        for layer in self.order:
            if layer == 'conv':
                self.ops_list.append(self.conv)
            elif layer == 'norm' and self.with_norm:
                self.ops_list.append(self.norm)
            elif layer == 'act' and self.with_activation:
                self.ops_list.append(self.activate)

        self.init_weights()

    def init_weights(self):
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.2)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        """Forward function."""
        for layer in self.ops_list:
            x = layer(x)
        return x
