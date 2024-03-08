# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Loss and time monitor definition."""
import os
import time
import numpy as np
from mindspore import Tensor
from mindspore.train.callback import Callback


class EvalCallBack(Callback):
    """Callback for inference while training."""
    def __init__(self, model, eval_dataset, eval_interval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_interval = eval_interval

    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_interval == 0:
            metrics = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            print("==== epoch: {:3d}, device id: {:2d}, top1-acc: {:1.4f}, top5-acc: {:1.4f}".format(
                cur_epoch, int(os.getenv('DEVICE_ID')), metrics['Top1-Acc'], metrics['Top5-Acc']), flush=True)


class TimeLossMonitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy array): train lr

    Returns:
        None

    Examples:
        >>> TimeLossMonitor(100,lr_init=Tensor([0.05]*100).asnumpy())
    """

    def __init__(self, lr_init=None):
        super(TimeLossMonitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)

    def epoch_begin(self, run_context):
        """Epoch begin."""
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """Epoch end."""
        cb_params = run_context.original_args()

        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / cb_params.batch_num
        print("epoch: [{:3d}/{:3d}], epoch time: {:5.3f}, steps: {:5d}, "
              "per step time: {:5.3f}, avg loss: {:5.3f}, lr:[{:5.3f}]".format(
                  cb_params.cur_epoch_num, cb_params.epoch_num, epoch_mseconds, cb_params.batch_num,
                  per_step_mseconds, np.mean(self.losses), self.lr_init[cb_params.cur_step_num - 1]), flush=True)

    def step_begin(self, run_context):
        """Step begin."""
        self.step_time = time.time()

    def step_end(self, run_context):
        """step end"""
        cb_params = run_context.original_args()
        step_loss = cb_params.net_outputs

        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        self.losses.append(step_loss)
