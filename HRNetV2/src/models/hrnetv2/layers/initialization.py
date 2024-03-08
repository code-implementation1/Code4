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
"""Initialization."""
import mindspore as ms
from mindspore import nn
from mindspore.common.initializer import (
    Constant, XavierUniform, Normal, TruncatedNormal, Uniform, HeNormal,
    HeUniform, initializer
)


def constant_init(module: nn.Cell, val: float, bias: float = 0) -> None:
    """Init module parameters by constant value."""
    w_init = Constant(val)
    b_init = Constant(bias)
    for name, param in module.parameters_and_names():
        new_tensor = None
        if name in ('weight', 'gamma'):
            new_tensor = initializer(w_init, param.data.shape, ms.float32)
        if name in ('bias', 'beta'):
            new_tensor = initializer(b_init, param.data.shape, ms.float32)
        if new_tensor is not None:
            param.data[:] = new_tensor


def xavier_init(module: nn.Cell, gain: float = 1, bias: float = 0) -> None:
    """Init module parameters by Xavier initialization."""
    w_init = XavierUniform(gain=gain)
    b_init = Constant(bias)
    for name, param in module.parameters_and_names():
        new_tensor = None
        if name in ('weight', 'gamma'):
            new_tensor = initializer(w_init, param.data.shape, ms.float32)
        if name in ('bias', 'beta'):
            new_tensor = initializer(b_init, param.data.shape, ms.float32)
        if new_tensor is not None:
            param.data[:] = new_tensor


def normal_init(
        module: nn.Cell, mean: float = 0, std: float = 1, bias: float = 0
) -> None:
    """Init module parameters by normal distribution."""
    w_init = Normal(mean=mean, sigma=std)
    b_init = Constant(bias)
    for name, param in module.parameters_and_names():
        new_tensor = None
        if name in ('weight', 'gamma'):
            new_tensor = initializer(w_init, param.data.shape, ms.float32)
        if name in ('bias', 'beta'):
            new_tensor = initializer(b_init, param.data.shape, ms.float32)
        if new_tensor is not None:
            param.data[:] = new_tensor


def trunc_normal_init(
        module: nn.Cell, std: float = 1, bias: float = 0
) -> None:
    """Init module parameters by truncated normal distribution."""
    w_init = TruncatedNormal(sigma=std)
    b_init = Constant(bias)
    for name, param in module.parameters_and_names():
        new_tensor = None
        if name in ('weight', 'gamma'):
            new_tensor = initializer(w_init, param.data.shape, ms.float32)
        if name in ('bias', 'beta'):
            new_tensor = initializer(b_init, param.data.shape, ms.float32)
        if new_tensor is not None:
            param.data[:] = new_tensor


def uniform_init(
        module: nn.Cell, a: float = 0, b: float = 1, bias: float = 0
) -> None:
    """Init module parameters by uniform distribution."""
    w_init = Uniform((b - a) / 2)
    b_init = Constant(bias)
    for name, param in module.parameters_and_names():
        new_tensor = None
        if name in ('weight', 'gamma'):
            new_tensor = initializer(w_init, param.data.shape, ms.float32)
        if name in ('bias', 'beta'):
            new_tensor = initializer(b_init, param.data.shape, ms.float32)
        if new_tensor is not None:
            param.data[:] = new_tensor
            param.data[:] += (b - a) / 2 + a


def kaiming_init(
        module: nn.Cell, a: float = 0, mode: str = 'fan_out',
        nonlinearity: str = 'relu', bias: float = 0,
        distribution: str = 'normal'
) -> None:
    """Init module parameters by the He initialization."""
    assert distribution in ['uniform', 'normal']
    if distribution == 'normal':
        w_init = HeNormal(
            negative_slope=a, mode=mode, nonlinearity=nonlinearity
        )
    else:
        w_init = HeUniform(
            negative_slope=a, mode=mode, nonlinearity=nonlinearity
        )
    b_init = Constant(bias)
    for name, param in module.parameters_and_names():
        new_tensor = None
        if name in ('weight', 'gamma'):
            new_tensor = initializer(w_init, param.data.shape, ms.float32)
        if name in ('bias', 'beta'):
            new_tensor = initializer(b_init, param.data.shape, ms.float32)
        if new_tensor is not None:
            param.data[:] = new_tensor
