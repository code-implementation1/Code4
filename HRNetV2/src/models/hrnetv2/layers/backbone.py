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
"""HRNet backbone and HRModule block."""
import mindspore as ms
from mindspore import nn, ops
from mindspore.nn import Identity

from .resnet import BasicBlock, Bottleneck


class ResizeNearest(nn.Cell):
    def __init__(self, scale=1.0):
        super(ResizeNearest, self).__init__()
        self.scale = scale

    def construct(self, x):
        size = (int(x.shape[-2] * self.scale),
                int(x.shape[-1] * self.scale))
        return ops.ResizeNearestNeighbor(size, align_corners=True)(x)


def pretty_describe(ar):
    return str(ar.reshape((-1,))[:10])


def pretty_list(ar):
    return pretty_describe(ar[0])


class HRModule(nn.Cell):
    """High-Resolution Module for HRNet.

    In this module, every branch has 4 BasicBlocks/Bottlenecks. Fusion/Exchange
    is in this module.
    """

    def __init__(self,
                 num_branches,
                 blocks,
                 num_blocks,
                 in_channels,
                 num_channels,
                 multiscale_output=True,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 block_init_cfg=None,
                 init_cfg=None,
                 norm_eval=False):
        super(HRModule, self).__init__(init_cfg)

        if norm_cfg is None:
            norm_cfg = dict(type='BN')

        self.block_init_cfg = block_init_cfg
        self._check_branches(num_branches, num_blocks, in_channels,
                             num_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches
        self.norm_eval = norm_eval

        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(num_branches, blocks, num_blocks,
                                            num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU()

    def _check_branches(self, num_branches, num_blocks, in_channels,
                        num_channels):
        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                        f'!= NUM_BLOCKS({len(num_blocks)})'
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                        f'!= NUM_CHANNELS({len(num_channels)})'
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                        f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.in_channels[branch_index] != \
                num_channels[branch_index] * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv2d(
                    self.in_channels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    pad_mode='valid',
                    has_bias=False),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion, momentum=0.9,
                    use_batch_statistics=False if self.norm_eval else None
                )
            ])

        layers = [block(
            self.in_channels[branch_index],
            num_channels[branch_index],
            stride,
            downsample=downsample,
            norm_eval=self.norm_eval
        )]
        self.in_channels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for _ in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index],
                    norm_eval=self.norm_eval
                ))

        return nn.SequentialCell(layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.CellList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.SequentialCell([
                            nn.Conv2d(
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                pad_mode='pad',
                                has_bias=False),
                            nn.BatchNorm2d(
                                in_channels[i], momentum=0.9,
                                use_batch_statistics=False
                                if self.norm_eval else None
                            ),
                            nn.Upsample(
                                scale_factor=2.0**(j - i), mode='nearest',
                                recompute_scale_factor=True
                            )
                        ])
                    )
                elif j == i:
                    fuse_layer.append(Identity())
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.SequentialCell([
                                    nn.Conv2d(
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        pad_mode='pad',
                                        has_bias=False),
                                    nn.BatchNorm2d(
                                        in_channels[i], momentum=0.9,
                                        use_batch_statistics=False
                                        if self.norm_eval else None
                                    )
                                ])
                            )
                        else:
                            conv_downsamples.append(
                                nn.SequentialCell([
                                    nn.Conv2d(
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        pad_mode='pad',
                                        has_bias=False),
                                    nn.BatchNorm2d(
                                        in_channels[j], momentum=0.9,
                                        use_batch_statistics=False
                                        if self.norm_eval else None
                                    ),
                                    nn.ReLU()
                                ])
                            )
                    fuse_layer.append(nn.SequentialCell(conv_downsamples))
            fuse_layers.append(nn.CellList(fuse_layer))

        return nn.CellList(fuse_layers)

    def construct(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = ms.Tensor(0)
            for j in range(self.num_branches):
                y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        return x_fuse


class HRNet(nn.Cell):
    """HRNet backbone.

    `High-Resolution Representations for Labeling Pixels and Regions
    arXiv: <https://arxiv.org/abs/1904.04514>`_.

    Args:
        extra (dict): Detailed configuration for each stage of HRNet.
            There must be 4 stages, the configuration for each stage must have
            5 keys:

                - num_modules(int): The number of HRModule in this stage.
                - num_branches(int): The number of branches in the HRModule.
                - block(str): The type of convolution block.
                - num_blocks(tuple): The number of blocks in each branch.
                  The length must be equal to num_branches.
                - num_channels(tuple): The number of channels in each branch.
                  The length must be equal to num_branches.
        in_channels (int): Number of input image channels. Default: 3.
        conv_cfg (dict): Dictionary to construct and config conv layer.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: False.
        multiscale_output (bool): Whether to output multi-level features
            produced by multiple branches. If False, only the first level
            feature will be output. Default: True.
        pretrained (str, optional): Model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Example:
        >>> from mmdet.models import HRNet
        >>> import torch
        >>> extra = dict(
        >>>     stage1=dict(
        >>>         num_modules=1,
        >>>         num_branches=1,
        >>>         block='BOTTLENECK',
        >>>         num_blocks=(4, ),
        >>>         num_channels=(64, )),
        >>>     stage2=dict(
        >>>         num_modules=1,
        >>>         num_branches=2,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4),
        >>>         num_channels=(32, 64)),
        >>>     stage3=dict(
        >>>         num_modules=4,
        >>>         num_branches=3,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4),
        >>>         num_channels=(32, 64, 128)),
        >>>     stage4=dict(
        >>>         num_modules=3,
        >>>         num_branches=4,
        >>>         block='BASIC',
        >>>         num_blocks=(4, 4, 4, 4),
        >>>         num_channels=(32, 64, 128, 256)))
        >>> self = HRNet(extra, in_channels=1)
        >>> self.eval()
        >>> inputs = torch.rand(1, 1, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 32, 8, 8)
        (1, 64, 4, 4)
        (1, 128, 2, 2)
        (1, 256, 1, 1)
    """

    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=None,
                 norm_eval=False,
                 with_cp=False,
                 multiscale_output=True):
        super(HRNet, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type='BN')

        # Assert whether the length of `num_blocks` and `num_channels` are
        # equal to `num_branches`
        for i in range(4):
            cfg = extra[f'stage{i + 1}']
            assert len(cfg['num_blocks']) == cfg['num_branches'] and \
                   len(cfg['num_channels']) == cfg['num_branches']

        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        # stem net
        self.bn1 = nn.BatchNorm2d(
            64, momentum=0.9,
            use_batch_statistics=False if self.norm_eval else None
        )
        self.bn2 = nn.BatchNorm2d(
            64, momentum=0.9,
            use_batch_statistics=False if self.norm_eval else None
        )

        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            pad_mode='pad',
            has_bias=False)

        self.conv2 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            pad_mode='pad',
            has_bias=False)

        self.relu = nn.ReLU()

        # stage 1
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self._make_layer(block, 64, num_channels, num_blocks)

        # stage 2
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition1 = self._make_transition_layer([stage1_out_channels],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        # stage 3
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        # stage 4
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multiscale_output=multiscale_output)

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.SequentialCell([
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                pad_mode='pad',
                                has_bias=False),
                            nn.BatchNorm2d(
                                num_channels_cur_layer[i], momentum=0.9,
                                use_batch_statistics=False
                                if self.norm_eval else None
                            ),
                            nn.ReLU()
                        ])
                    )
                else:
                    transition_layers.append(Identity())
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.SequentialCell([
                            nn.Conv2d(
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                pad_mode='pad',
                                has_bias=False),
                            nn.BatchNorm2d(
                                out_channels, momentum=0.9,
                                use_batch_statistics=False
                                if self.norm_eval else None
                            ),
                            nn.ReLU()
                        ])
                    )
                transition_layers.append(nn.SequentialCell(conv_downsamples))

        return nn.CellList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.SequentialCell([
                nn.Conv2d(
                    inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    pad_mode='valid',
                    has_bias=False),
                nn.BatchNorm2d(
                    planes * block.expansion, momentum=0.9,
                    use_batch_statistics=False if self.norm_eval else None
                )
            ])

        layers = []
        layers.append(
            block(
                inplanes,
                planes,
                stride,
                downsample=downsample,
            ))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    inplanes,
                    planes,
                ))

        return nn.SequentialCell(layers)

    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        hr_modules = []
        block_init_cfg = None

        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                HRModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    block_init_cfg=block_init_cfg,
                    norm_eval=self.norm_eval))

        return nn.SequentialCell(hr_modules), in_channels

    def construct(self, x):
        """Forward function."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            x_list.append(self.transition1[i](x))
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if not isinstance(self.transition2[i], Identity):
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if not isinstance(self.transition3[i], Identity):
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        return y_list
