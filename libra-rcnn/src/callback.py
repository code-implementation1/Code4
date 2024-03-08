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
"""Custom training and evaluation callbacks."""
import time

import logging
import mindspore as ms
from mindspore import Callback, SummaryCollector, SummaryRecord, RunContext
from .mlflow_funcs import mlflow_import


class EvalCocoCallback(Callback):
    """
    Evaluation callback when training.
    """

    def __init__(
            self, best_ckpt_path, buffer=5, prefix='model'
    ):
        super().__init__()
        self.best_epoch = 0
        self.best_res = 0
        self.best_ckpt_path = best_ckpt_path
        self.prefix = prefix
        self.buffer_size = buffer
        self.current_ckpt = []
        self._make_dir()

    def on_eval_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        res = cb_params.eval_results['mAP']
        network = cb_params.network
        if res > self.best_res:
            self.best_epoch = cur_epoch
            self.best_res = res
            self._save_checkpoint(
                network, epoch=cb_params.cur_epoch_num
            )

    def _save_checkpoint(self, network, epoch):
        """
        Save checkpoint.

        Parameters
        ----------
        network
            Network to save checkpoint for.
        """
        # TODO: May not work with model arts or distributed training.
        if not float('-inf') < self.best_res < float('inf'):
            return
        ckpt_name = f'epoch={epoch}_mAP={self.best_res:.3f}.ckpt'
        if self.prefix:
            ckpt_name = f'{self.prefix}_{ckpt_name}'
        file_path = self.best_ckpt_path / ckpt_name
        ms.save_checkpoint(network, str(file_path))

        # mlflow = mlflow_import()
        # if mlflow is not None and mlflow.active_run() is not None:
        #     mlflow.log_artifact(str(file_path))

        self.current_ckpt.append(file_path)
        if len(self.current_ckpt) > self.buffer_size:
            removed = self.current_ckpt[0]
            removed.unlink()
            del self.current_ckpt[0]

    def _make_dir(self):
        """Create a checkpoint directory."""
        if not self.best_ckpt_path.exists():
            self.best_ckpt_path.mkdir(parents=True)
            logging.info('Directory created: %s', str(self.best_ckpt_path))
        else:
            logging.warning('Directory already exists: %s',
                            str(self.best_ckpt_path))


class SummaryCallbackWithEval(SummaryCollector):
    """
    Callback that can collect a common information like SummaryCollector.

    Additionally, this callback collects:
        - learning rate
        - validation loss
        - validation accuracy
    """

    def __init__(
            self,
            summary_dir,
            logs_dir,
            collect_freq=10,
            collect_specified_data=None,
            keep_default_action=True,
            custom_lineage_data=None,
            collect_tensor_freq=None,
            max_file_size=None,
            export_options=None,
            print_loss_every=1,
    ):
        super().__init__(
            str(summary_dir),
            collect_freq,
            collect_specified_data,
            keep_default_action,
            custom_lineage_data,
            collect_tensor_freq,
            max_file_size,
            export_options
        )
        self.entered_count = 0
        self.logs_dir = logs_dir
        self.print_loss_every = print_loss_every
        self.init_loss_monitoring()

    def init_loss_monitoring(self):
        self.loss_sum = 0
        self.total_loss = 0
        self.steps = 0
        self.total_steps = 0

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs.asnumpy()
        cur_step_in_epoch = (
            (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        )

        self.steps += 1
        self.total_steps += 1
        self.loss_sum += float(loss)
        self.total_loss += float(loss)
        lr = self.get_learning_rate(run_context)
        if self.steps == self.print_loss_every:
            loss = self.loss_sum / self.steps
            logging.info('epoch: %i, step: %i, loss: %f,  lr: %f',
                         cb_params.cur_epoch_num, cur_step_in_epoch, loss, lr)

            self.steps = 0
            self.loss_sum = 0

    def on_train_epoch_end(self, run_context: RunContext):
        """
        Collect learning rate after train epoch.

        Parameters
        ----------
        run_context: RunContext
        """
        cb_params = run_context.original_args()
        lr = self.get_learning_rate(run_context)
        if lr is not None:
            self._record.add_value(
                'scalar', 'Train/learning_rate', ms.Tensor(lr)
            )
            loss = self.total_loss / self.total_steps
            self._record.add_value('scalar', 'Train/loss',
                                   ms.Tensor(loss).mean())
            self.init_loss_monitoring()
            self._record.record(cb_params.cur_epoch_num)
            super().on_train_epoch_end(run_context)

            mlflow = mlflow_import()
            if mlflow is not None and mlflow.active_run() is not None:
                mlflow.log_metric('Train/learning_rate', lr,
                                  cb_params.cur_epoch_num)
                mlflow.log_metric('Train/loss', loss, cb_params.cur_epoch_num)

    def on_eval_end(self, run_context: RunContext):
        """
        Collect metrics after evaluation complete.

        Parameters
        ----------
        run_context: RunContext
        """
        cb_params = run_context.original_args()
        metrics = {k: v for k, v in cb_params.eval_results.items() if v >= 0}

        mlflow = mlflow_import()
        for metric_name, value in metrics.items():
            self._record.add_value(
                'scalar', f'Metrics/{metric_name}', ms.Tensor(value)
            )
            if mlflow is not None and mlflow.active_run() is not None:
                mlflow.log_metric(f'Metrics/{metric_name}', value,
                                  cb_params.cur_epoch_num)

        metrics['loss'] = self.total_loss / self.total_steps
        logging.info(
            'Result metrics for epoch %i: %s', cb_params.cur_epoch_num,
            str({key: metrics[key] for key in sorted(metrics)})
        )

        self._record.record(cb_params.cur_epoch_num)
        self._record.flush()

        if mlflow is not None and mlflow.active_run() is not None:
            mlflow.log_artifacts(self.logs_dir)

    def __enter__(self):
        """
        Enter in context manager and control that SummaryRecord created once.
        """
        if self.entered_count == 0:
            self._record = SummaryRecord(log_dir=self._summary_dir,
                                         max_file_size=self._max_file_size,
                                         raise_exception=False,
                                         export_options=self._export_options)
            self._first_step, self._dataset_sink_mode = True, True
        self.entered_count += 1
        return self

    def __exit__(self, *err):
        """
        Exit from context manager and control SummaryRecord correct closing.
        """
        self.entered_count -= 1
        if self.entered_count == 0:
            super().__exit__(err)

    def get_learning_rate(self, run_context):
        cb_params = run_context.original_args()
        optimizer = cb_params.get('optimizer')
        if optimizer is None:
            optimizer = getattr(cb_params.network, 'optimizer')
        lr = None
        if optimizer is None:
            logging.warning('There is no optimizer found!')
        else:
            global_step = optimizer.global_step
            lr = optimizer.learning_rate(global_step)[0]
        return lr


class TrainTimeMonitor(Callback):
    """
    Monitor the time in train process.

    Parameters
    ----------
    data_size: int
        How many steps are the intervals between print information each
        time. If the program get `batch_num` during training,
        `data_size` will be set to `batch_num`, otherwise
        `data_size` will be used. Default: None

    Raises
    ------
        ValueError: If data_size is not positive int.
    """

    def __init__(self, data_size=None):
        super().__init__()
        self.data_size = data_size
        self.epoch_time = time.time()

    def on_train_epoch_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Parameters
        ----------
        run_context: RunContext
            Context of the process running. For more details, please refer to
            :class:`mindspore.RunContext`
        """
        self.epoch_time = time.time()

    def on_train_epoch_end(self, run_context):
        """
        Log process cost time at the end of epoch.

        Parameters
        ----------
        run_context: RunContext
            Context of the process running. For more details, please refer to
            :class:`mindspore.RunContext`
        """
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        step_size = self.data_size
        cb_params = run_context.original_args()
        mode = cb_params.get('mode', '')
        if hasattr(cb_params, 'batch_num'):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num

        step_seconds = epoch_seconds / step_size
        logging.info('%s epoch time: %5.3f ms, per step time: %5.3f ms',
                     mode.title(), epoch_seconds, step_seconds)


class EvalTimeMonitor(Callback):
    """
    Monitor the time in eval process.

    Parameters
    ----------
    data_size: int
        How many steps are the intervals between print information each
        time. If the program get `batch_num` during training,
        `data_size` will be set to `batch_num`, otherwise
        `data_size` will be used. Default: None


    Raises
    ------
        ValueError: If data_size is not positive int.
    """

    def __init__(self, data_size=None):
        super().__init__()
        self.data_size = data_size
        self.epoch_time = time.time()

    def on_eval_epoch_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Parameters
        ----------
        run_context:
            Context of the process running. For more details, please refer to
            :class:`mindspore.RunContext`
        """
        self.epoch_time = time.time()

    def on_eval_epoch_end(self, run_context):
        """
        Log process cost time at the end of epoch.

        Parameters
        ----------
        run_context:
            Context of the process running. For more details, please refer to
            :class:`mindspore.RunContext`
        """
        epoch_seconds = (time.time() - self.epoch_time) * 1000
        step_size = self.data_size
        cb_params = run_context.original_args()
        mode = cb_params.get('mode', '')
        if hasattr(cb_params, 'batch_num'):
            batch_num = cb_params.batch_num
            if isinstance(batch_num, int) and batch_num > 0:
                step_size = cb_params.batch_num

        step_seconds = epoch_seconds / step_size
        logging.info('%s epoch time: %5.3f ms, per step time: %5.3f ms',
                     mode.title(), epoch_seconds, step_seconds)
