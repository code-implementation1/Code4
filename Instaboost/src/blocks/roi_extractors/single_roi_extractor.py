# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
"""RoI extractor module."""
import os
from typing import Sequence

import numpy as np
import mindspore as ms

from mindspore import ops
from mindspore import nn
from mindspore.nn import layer as L
from mindspore.common.tensor import Tensor

from .. import Config


class ROIAlign(nn.Cell):
    """Custom ROIAlign based on mmcv implementation

    Args:
        pooled_height (int): Output height.
        pooled_width (int): Output width.
        spatial_scale (float): Scale the input boxes by this number
        sampling_ratio (int): Number of inputs samples to take for each
            output sample. 0 to take samples densely for current models.
        pool_mode (str): 'avg' or 'max'. Pooling mode in each bin.
        aligned (bool): if False, use the legacy implementation in
            MMDetection. If True, align the results more perfectly.
    """

    def __init__(
            self,
            pooled_height: int = 7,
            pooled_width: int = 7,
            spatial_scale: float = 1.0,
            sampling_ratio: float = 0,
            pool_mode: str = 'avg',
            aligned: bool = True
    ):
        """Init ROIAlign"""
        super(ROIAlign, self).__init__()
        self.pooled_height = ms.Tensor(pooled_height, ms.int32)
        self.pooled_width = ms.Tensor(pooled_width, ms.int32)
        self.spatial_scale = ms.Tensor(spatial_scale, ms.float32)
        self.sampling_ratio = ms.Tensor(sampling_ratio, ms.int32)

        if pool_mode == 'avg':
            self.pool_mode = ms.Tensor(1, ms.int32)
        elif pool_mode == 'max':
            self.pool_mode = ms.Tensor(0, ms.int32)
        else:
            raise ValueError(
                f'pool_mode must be in{("avg", "max")} but {pool_mode} was '
                f'obtained.'
            )

        self.aligned = ms.Tensor(1 if aligned else 0, ms.int32)

        reg_info = ops.CustomRegOp(op_name="RoIAlign") \
            .input(0, 'features') \
            .input(1, 'rois') \
            .input(2, 'pooled_height') \
            .input(3, 'pooled_width') \
            .input(4, 'spatial_scale') \
            .input(5, 'sampling_ratio') \
            .input(6, 'pool_mode') \
            .input(7, 'aligned') \
            .output(0, 'roi_feats') \
            .output(1, 'argmax_y') \
            .output(2, 'argmax_x') \
            .dtype_format(
                ops.DataType.None_None, ops.DataType.None_None,
                ops.DataType.None_None, ops.DataType.None_None,
                ops.DataType.None_None, ops.DataType.None_None,
                ops.DataType.None_None, ops.DataType.None_None,
                ops.DataType.None_None, ops.DataType.None_None,
                ops.DataType.None_None
            ).target('GPU').get_op_info()

        self.forward_op = ops.Custom(
            os.path.join(
                os.path.dirname(__file__), 'bin', 'roi_align.so:CustomROIAlign'
            ),
            out_shape=self._get_out_shape,
            out_dtype=self._get_out_type,
            bprop=self.backward(),
            func_type='aot',
            reg_info=reg_info
        )

    def backward(self):
        """Prepare backward."""
        reg_info = ops.CustomRegOp() \
            .input(0, 'features') \
            .input(1, 'rois') \
            .input(2, 'argmax_y') \
            .input(3, 'argmax_x') \
            .input(4, 'grad_output') \
            .input(5, 'pooled_height') \
            .input(6, 'pooled_width') \
            .input(7, 'spatial_scale') \
            .input(8, 'sampling_ratio') \
            .input(9, 'pool_mode') \
            .input(10, 'aligned') \
            .output(0, 'grad_input') \
            .dtype_format(
                ops.DataType.None_None, ops.DataType.None_None,
                ops.DataType.None_None, ops.DataType.None_None,
                ops.DataType.None_None, ops.DataType.None_None,
                ops.DataType.None_None, ops.DataType.None_None,
                ops.DataType.None_None, ops.DataType.None_None,
                ops.DataType.None_None, ops.DataType.None_None
            ).target('GPU').get_op_info()

        backward_op = ops.Custom(
            os.path.join(
                os.path.dirname(__file__),
                'bin', 'roi_align.so:CustomROIAlignBackward'
            ),
            self._get_out_shape_backward,
            self._get_out_type_backward,
            func_type="aot",
            reg_info=reg_info
        )

        def custom_bprop(
                features,
                rois,
                pooled_height,
                pooled_width,
                spatial_scale,
                sampling_ratio,
                pool_mode,
                aligned,
                outputs,
                dout
        ):
            outputs, argmax_y, argmax_x = outputs
            grad_output = dout[0]
            df = backward_op(
                features,
                rois,
                argmax_y,
                argmax_x,
                grad_output,
                pooled_height,
                pooled_width,
                spatial_scale,
                sampling_ratio,
                pool_mode,
                aligned,
            )
            dr = ops.zeros_like(rois)
            return df, dr, 0., 0., 0., 0., 0., 0.

        return custom_bprop

    def _get_out_shape(
            self, inputs, rois_, *x,
    ):
        return (
            (rois_[0], inputs[1], self.pooled_height, self.pooled_width),
            (rois_[0], inputs[1], self.pooled_height, self.pooled_width),
            (rois_[0], inputs[1], self.pooled_height, self.pooled_width),
        )

    @staticmethod
    def _get_out_type(inputs, rois_, *x):
        return ms.float32, ms.float32, ms.float32

    @staticmethod
    def _get_out_shape_backward(features, *x):
        return features

    @staticmethod
    def _get_out_type_backward(features, *x):
        return features

    def construct(self, features: ms.Tensor, rois: ms.Tensor) -> ms.Tensor:
        """Forward roi align."""
        roi_feats, _, _ = self.forward_op(
            features,
            rois,
            self.pooled_height,
            self.pooled_width,
            self.spatial_scale,
            self.sampling_ratio,
            self.pool_mode,
            self.aligned
        )
        return roi_feats


class SingleRoIExtractor(nn.Cell):
    """
    Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale.

    Args:
        roi_layer (Config): Specify RoI layer type and arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (Sequence[int]): Strides of input feature maps.
        train_batch_size (int): Batch size in training mode.
        test_batch_size (int): Batch size in test mode.
        finest_scale (int): Scale threshold of mapping to level 0.

    Examples:
        >>> roi_layer = Config(dict(out_size=7, sampling_ratio=0))
        >>> out_channels=256
        >>> featmap_strides=[4, 8, 16, 32]
        >>> finest_scale=56
        >>> train_bs, test_bs = 640, 1000
        >>> feature_shapes = [[256 // 2 ** i] * 2 for i in range(4)]
        >>> features = [
        ...    ms.Tensor(np.random.randn(1, 256, *fn), ms.float32)
        ...    for fn in feature_shapes
        ... ]
        >>> roi_extractor = SingleRoIExtractor(
        ...     roi_layer=roi_layer, out_channels=out_channels,
        ...     featmap_strides=featmap_strides, train_batch_size=train_bs,
        ...     test_batch_size=test_bs, finest_scale=finest_scale
        ... )
        >>> rois_train = np.random.uniform(0, 1024, (train_bs, 4))
        >>> rois_train[:, [2, 3]] += rois_train[:, [0, 1]]
        >>> rois_train = np.concatenate((np.ones((train_bs, 1)), rois_train), axis=1)
        >>> rois_train = ms.Tensor(rois_train / 2, ms.float32)
        >>> roi_feats = roi_extractor(rois_train, features)
        >>> roi_feats.shape
        (640, 256, 7, 7)
        >>> roi_extractor.set_train_local(False)
        >>> rois_infer = np.random.uniform(0, 1024, (test_bs, 4))
        >>> rois_infer[:, [2, 3]] += rois_infer[:, [0, 1]]
        >>> rois_infer = np.concatenate((np.ones((test_bs, 1)), rois_infer), axis=1)
        >>> rois_infer = ms.Tensor(rois_infer / 2, ms.float32)
        >>> roi_feats = roi_extractor(rois_infer, features)
        >>> roi_feats.shape
        (1000, 256, 7, 7)
    """

    def __init__(
            self,
            roi_layer: Config,
            out_channels: int,
            featmap_strides: Sequence[int],
            train_batch_size: int = 640 * 4,
            test_batch_size: int = 1000 * 1,
            finest_scale: int = 56
    ):
        """Init SingleRoIExtractor."""
        super(SingleRoIExtractor, self).__init__()
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.out_channels = out_channels
        self.featmap_strides = featmap_strides
        self.num_levels = len(self.featmap_strides)
        self.out_size = roi_layer.out_size
        self.sampling_ratio = roi_layer.sampling_ratio
        self.roi_layers = self.build_roi_layers(self.featmap_strides)
        self.roi_layers = L.CellList(self.roi_layers)

        self.finest_scale_ = finest_scale

        self.dtype = np.float32
        self.ms_dtype = ms.float32
        self.set_train_local(training=True)

    def set_train_local(self, training: bool = True):
        """Set training flag."""
        self.training_local = training

        # Init tensor
        self.batch_size = (
            self.train_batch_size if self.training_local
            else self.test_batch_size
        )
        self.ones = Tensor(
            np.array(np.ones((self.batch_size, 1)), dtype=self.dtype)
        )
        finest_scale = np.array(
            np.ones((self.batch_size, 1)), dtype=self.dtype
        ) * self.finest_scale_
        self.finest_scale = Tensor(finest_scale)
        self.epslion = Tensor(
            np.array(np.ones((self.batch_size, 1)), dtype=self.dtype) *
            self.dtype(1e-6)
        )
        self.zeros = Tensor(
            np.array(np.zeros((self.batch_size, 1)), dtype=np.int32)
        )
        self.max_levels = Tensor(
            np.array(np.ones((self.batch_size, 1)), dtype=np.int32) *
            (self.num_levels - 1)
        )
        self.res_ = Tensor(
            np.array(
                np.zeros((self.batch_size, self.out_channels, self.out_size,
                          self.out_size)),
                dtype=self.dtype
            )
        )

    def num_inputs(self):
        """Number of input feature maps."""
        return len(self.featmap_strides)

    def build_roi_layers(
            self, featmap_strides: Sequence[int]
    ) -> Sequence[ms.nn.Cell]:
        """Build roi_align layers."""
        roi_layers = []
        for s in featmap_strides:
            layer_cls = ROIAlign(pooled_width=self.out_size,
                                 pooled_height=self.out_size,
                                 spatial_scale=1 / s,
                                 sampling_ratio=self.sampling_ratio)
            roi_layers.append(layer_cls)
        return roi_layers

    def map_roi_levels(self, rois: ms.Tensor) -> ms.Tensor:
        """Map rois to corresponding feature levels by scales.

        - scale < finest_scale * 2: level 0
        - finest_scale * 2 <= scale < finest_scale * 4: level 1
        - finest_scale * 4 <= scale < finest_scale * 8: level 2
        - scale >= finest_scale * 8: level 3

        Args:
            rois (Tensor): Input RoIs, Shape (k, 5).

        Returns:
            Tensor: Level index (0-based) of each RoI, shape (k, )
        """
        scale = ops.sqrt(
            (rois[:, 3:4] - rois[:, 1:2]) * (rois[:, 4:5] - rois[:, 2:3])
        )
        target_lvls = ops.log2(scale / self.finest_scale + self.epslion)
        target_lvls = ops.floor(target_lvls)
        target_lvls = ops.cast(target_lvls, ms.int32)
        target_lvls = ops.clip_by_value(
            target_lvls, clip_value_min=self.zeros,
            clip_value_max=self.max_levels
        )
        return target_lvls

    def roi_rescale(self, rois: ms.Tensor, scale_factor: float) -> ms.Tensor:
        """Scale RoI coordinates by scale factor.
        Args:
            rois (ms.Tensor): RoI (Region of Interest), shape (n, 5)
            scale_factor (float): Scale factor that RoI will be multiplied by.

        Returns:
            Tensor: Scaled RoI.
        """
        cx = (rois[:, 1] + rois[:, 3]) * 0.5
        cy = (rois[:, 2] + rois[:, 4]) * 0.5
        w = rois[:, 3] - rois[:, 1]
        h = rois[:, 4] - rois[:, 2]
        new_w = w * scale_factor
        new_h = h * scale_factor
        x1 = cx - new_w * 0.5
        x2 = cx + new_w * 0.5
        y1 = cy - new_h * 0.5
        y2 = cy + new_h * 0.5
        new_rois = ops.stack((rois[:, 0], x1, y1, x2, y2), axis=-1)
        return new_rois

    def construct(
            self, rois: ms.Tensor, feats: Sequence[ms.Tensor],
            roi_scale_factor: float = None
    ) -> ms.Tensor:
        """Extract features.

        Args:
            rois (ms.Tensor): Proposed bounding bboxes. Shape (k, 4).
            feats (Sequence[ms.Tensor]): List of features that are RoIs
                features sources. Length of list is len(self.featmap_strides).
            roi_scale_factor (float): Parameter that is used to scale input
                RoIs.

        Returns:
            ms.Tensor: Extracted RoIs. Shape
                (self.batch_size, self.out_channels, self.out_size, self.out_size).
        """
        res = self.res_
        target_lvls = self.map_roi_levels(rois)
        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)
        for i in range(self.num_levels):
            mask = ops.equal(target_lvls, ops.scalar_to_tensor(i, ms.int32))
            mask = ops.reshape(mask, (-1, 1, 1, 1))
            roi_feats_t = self.roi_layers[i](feats[i], rois)
            mask = ops.cast(
                ops.tile(
                    ops.cast(mask, ms.int32),
                    (1, self.out_channels, self.out_size, self.out_size)
                ),
                ms.bool_
            )
            res = ops.select(mask, roi_feats_t, res)

        return res
