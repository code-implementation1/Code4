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
"""Balance L1 regression loss."""
from typing import Optional

import mindspore as ms
from mindspore import nn
from mindspore import ops


class BalancedL1Loss(nn.Cell):
    """Balanced L1 Loss.
    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    Args:
        alpha (float): The denominator ``alpha`` in the balanced L1 loss.
            Defaults to 0.5.
        gamma (float): The ``gamma`` in the balanced L1 loss. Defaults to 1.5.
        beta (float, optional): The loss is a piecewise function of prediction
            and target. ``beta`` serves as a threshold for the difference
            between the prediction and target. Defaults to 1.0.
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(
            self,
            alpha: float = 0.5,
            gamma: float = 1.5,
            beta: float = 1.0,
            reduction: str = 'mean',
            loss_weight: float = 1.0
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = ms.Tensor(gamma, ms.float32)
        self.beta = ms.Tensor(beta, ms.float32)
        self.reduction = reduction
        self.loss_weight = loss_weight

    def construct(
            self, pred: ms.Tensor, target: ms.Tensor, weight: ms.Tensor
    ) -> ms.Tensor:
        """Forward function of loss.

        Args:
            pred (ms.Tensor): The prediction with shape (N, 4).
            target (ms.Tensor): The learning target of the prediction
                with shape (N, 4).
            weight (ms.Tensor): Sample-wise loss weight with
                shape (N, ).

        Returns:
            ms.Tensor: The calculated loss
        """
        loss_bbox = self.loss_weight * self.balanced_l1_loss(
            pred,
            target,
            weight=weight,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            reduction=self.reduction
        )
        return loss_bbox

    @staticmethod
    def balanced_l1_loss(
            pred: ms.Tensor,
            target: ms.Tensor,
            weight: Optional[ms.Tensor] = None,
            beta: ms.Tensor = ms.Tensor(1.0),
            alpha: float = 0.5,
            gamma: ms.Tensor = ms.Tensor(1.5),
            reduction: str = 'mean'
    ) -> ms.Tensor:
        """Calculate balanced L1 loss.
        Please see the `Libra R-CNN <https://arxiv.org/pdf/1904.02701.pdf>`_
        Args:
            pred (ms.Tensor): The prediction with shape (N, 4).
            target (ms.Tensor): The learning target of the prediction
                with shape (N, 4).
            weight (Optional[ms.Tensor]): Sample-wise loss weight with
                shape (N, ).
            beta (ms.Tensor): The loss is a piecewise function of
                prediction and target and ``beta`` serves as a threshold for
                the difference between the prediction and target.
                Defaults to 1.0.
            alpha (float): The denominator ``alpha`` in the balanced L1 loss.
                Defaults to 0.5.
            gamma (ms.Tensor): The ``gamma`` in the balanced L1 loss.
                Defaults to 1.5.
            reduction (str): Type of loss accumulation.
        Returns:
            ms.Tensor: The calculated loss
        """
        assert beta > 0

        diff = ops.abs(pred - target)
        b = ops.exp(gamma / alpha) - 1

        loss = ops.select(
            diff < beta,
            alpha / b * (b * diff + 1) *
            ops.log(b * diff / beta + 1) - alpha * diff,
            gamma * diff + gamma / b - alpha * beta
        )

        if weight is not None:
            loss = loss * weight

        if reduction == 'mean':
            if weight is not None:
                w_sum = ops.reduce_sum(weight)
                loss = ops.reduce_sum(loss)
                loss = loss / w_sum
            else:
                loss = ops.mean(loss)
        elif reduction == 'sum':
            loss = ops.reduce_sum(loss)

        return loss
