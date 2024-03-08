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
"""Blocks training network wrapper."""
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer


class LossNet(nn.Cell):
    """Blocks loss method."""

    def construct(
            self, rpn_loss, rcnn_loss, seg_loss, rpn_cls_loss, rpn_reg_loss,
            rcnn_cls_loss, rcnn_reg_loss
    ):
        return rpn_loss + rcnn_loss + seg_loss


class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to compute loss.

    Args:
        backbone (Cell): The target network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
    """

    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def set_train(self, mode: bool = True):
        super().set_train(mode=mode)
        self._backbone.set_train(mode)

    def construct(
            self, x, img_shape, gt_bboxes, gt_seg_masks, gt_label, gt_num
    ):
        (
            rpn_loss, rcnn_loss, seg_loss, rpn_cls_loss, rpn_reg_loss,
            rcnn_cls_loss, rcnn_reg_loss
        ) = self._backbone(
            x, img_shape, gt_bboxes, gt_seg_masks, gt_label, gt_num
        )

        return self._loss_fn(
            rpn_loss, rcnn_loss, seg_loss, rpn_cls_loss, rpn_reg_loss,
            rcnn_cls_loss, rcnn_reg_loss
        )

    @property
    def backbone_network(self):
        """
        Get the backbone network.

        Returns:
            Cell, return backbone network.
        """
        return self._backbone


_grad_scale = ops.MultitypeFuncGraph('grad_scale')


@_grad_scale.register('Tensor', 'Tensor')
def tensor_grad_scale(scale, grad):
    return grad * ops.cast(ops.Reciprocal()(scale), ops.dtype(grad))


class TrainOneStepCell(nn.TrainOneStepWithLossScaleCell):
    """Network training package class.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        grad_clip (bool): Whether clip gradients. Default value is False.
    """

    def __init__(self, network, optimizer, scale_sense=1, grad_clip=0.):
        if isinstance(scale_sense, (int, float)):
            scale_sense = ms.Tensor(scale_sense, ms.float32)
        super(TrainOneStepCell, self).__init__(network, optimizer, scale_sense)
        self.grad_clip = grad_clip

    def set_train(self, mode: bool = True):
        super().set_train(mode=mode)
        self.network.set_train(mode)

    def construct(
            self, x, img_shape, gt_bboxes, gt_seg_masks, gt_label, gt_num
    ):
        weights = self.weights
        loss = self.network(
            x, img_shape, gt_bboxes, gt_seg_masks, gt_label, gt_num
        )
        scaling_sens = self.scale_sense

        _, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = ops.ones_like(loss) * ops.cast(
            scaling_sens, ops.dtype(loss)
        )
        grads = self.grad(self.network, weights)(
            x, img_shape, gt_bboxes, gt_seg_masks, gt_label, gt_num,
            scaling_sens_filled
        )
        grads = self.hyper_map(ops.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        if self.grad_clip > 0.:
            grads = tuple([
                ops.clip_by_value(
                    g, clip_value_min=-self.grad_clip,
                    clip_value_max=self.grad_clip
                )
                for g in grads
            ])
            grads = ops.clip_by_global_norm(grads)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss


class TrainOneStepCellCPU(nn.Cell):
    """
    Network training package class.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network.
        optimizer (Cell): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default value is 1.0.
        reduce_flag (bool): The reduce flag. Default value is False.
        mean (bool): All reduce method. Default value is False.
        degree (int): Device number. Default value is None.
    """

    def __init__(
            self, network, optimizer, sens=1.0, reduce_flag=False, mean=True,
            degree=None
    ):
        super(TrainOneStepCellCPU, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ms.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = ops.GradOperation(get_by_list=True,
                                      sens_param=True)
        self.sens = ms.Tensor([sens,], ms.float32)
        self.reduce_flag = reduce_flag
        if reduce_flag:
            self.grad_reducer = DistributedGradReducer(
                optimizer.parameters, mean, degree
            )

    def set_train(self, mode: bool = True):
        super().set_train(mode=mode)
        self.network.set_train(mode)

    def construct(self, x, img_shape, gt_bboxe, gt_label, gt_num):
        weights = self.weights
        loss = self.network(x, img_shape, gt_bboxe, gt_label, gt_num)
        grads = self.grad(self.network, weights)(
            x, img_shape, gt_bboxe, gt_label, gt_num, self.sens
        )
        if self.reduce_flag:
            grads = self.grad_reducer(grads)

        return ops.depend(loss, self.optimizer(grads))
