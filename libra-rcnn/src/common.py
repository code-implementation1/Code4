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
"""Common utilities."""

import os
import logging

from mindspore import CheckpointConfig, ModelCheckpoint

from .mlflow_funcs import _get_rank
from .callback import (
    SummaryCallbackWithEval, EvalCocoCallback, TrainTimeMonitor,
    EvalTimeMonitor
)


def config_logging(
        filename_prefix, filemode='a', level=logging.INFO,
        log_format='[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S'
):
    """
    Configure logging.
    """
    rank = _get_rank()
    filename_suffix = '.log'
    if rank is not None:
        filename_suffix = f'_{rank}' + filename_suffix
    filename = filename_prefix + filename_suffix
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logging.basicConfig(level=level, datefmt=datefmt, format=log_format)
    file_handler = logging.FileHandler(filename, filemode)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt))
    file_handler.setLevel(level)
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(level)


def get_callbacks(
        arch, train_data_size, val_data_size, summary_dir, logs_dir,
        ckpt_dir, best_ckpt_dir, rank=0, ckpt_save_every_step=0,
        ckpt_save_every_sec=0, ckpt_keep_num=10, best_ckpt_num=5,
        print_loss_every=1, collect_freq=0, collect_tensor_freq=None,
        collect_graph=False, collect_input_data=False,
        keep_default_action=False
):
    """
    Get common callbacks.
    """
    if collect_freq == 0:
        collect_freq = train_data_size
    if ckpt_save_every_step == 0 and ckpt_save_every_sec == 0:
        ckpt_save_every_step = train_data_size
    prefix = f'{arch}_{rank}'

    config_ck = CheckpointConfig(
        save_checkpoint_steps=ckpt_save_every_step,
        save_checkpoint_seconds=ckpt_save_every_sec,
        keep_checkpoint_max=ckpt_keep_num,
        append_info=['epoch_num', 'step_num']
    )
    train_time_cb = TrainTimeMonitor(data_size=train_data_size)
    eval_time_cb = EvalTimeMonitor(data_size=val_data_size)
    best_ckpt_save_cb = EvalCocoCallback(
        best_ckpt_path=best_ckpt_dir, buffer=best_ckpt_num, prefix=prefix
    )

    ckpoint_cb = ModelCheckpoint(
        prefix=prefix,
        directory=str(ckpt_dir),
        config=config_ck
    )

    specified = {
        'collect_metric': True,
        'collect_train_lineage': True,
        'collect_eval_lineage': True,
        # "histogram_regular": "^network.*weight.*",
        'collect_graph': collect_graph,
        # "collect_dataset_graph": True,
        'collect_input_data': collect_input_data
    }
    summary_collector_cb = SummaryCallbackWithEval(
        summary_dir=summary_dir,
        logs_dir=logs_dir,
        collect_specified_data=specified,
        collect_freq=collect_freq,
        keep_default_action=keep_default_action,
        collect_tensor_freq=collect_tensor_freq,
        print_loss_every=print_loss_every
    )

    return [
        train_time_cb,
        eval_time_cb,
        ckpoint_cb,
        best_ckpt_save_cb,
        summary_collector_cb
    ]
